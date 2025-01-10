#include "spirv_generator.h"
#include "types.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/DerivedTypes.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"  // Add this include


namespace cspir {

void SPIRVGenerator::initializeModule() {
    Module = std::make_unique<llvm::Module>("spir_kernel", *LLVMCtx);
    Module->setTargetTriple("spir64-unknown-unknown");
}
llvm::FunctionCallee SPIRVGenerator::getOpenCLFunction(
    const std::string& Name,
    llvm::Type* RetTy,
    llvm::ArrayRef<llvm::Type*> ArgTypes) {

    return Module->getOrInsertFunction(
        Name,
        llvm::FunctionType::get(RetTy, ArgTypes, false)
    );
}

void SPIRVGenerator::addBarrier(unsigned Fence) {
    auto BarrierFn = getOpenCLFunction(
        "barrier",
        llvm::Type::getVoidTy(Builder.getContext()),
        {llvm::Type::getInt32Ty(Builder.getContext())}
    );

    Builder.CreateCall(BarrierFn, {
        llvm::ConstantInt::get(Builder.getContext(),
            llvm::APInt(32, Fence))
    });
}

llvm::Value* SPIRVGenerator::performVectorReduction(llvm::Value* Vec, unsigned Width) {
    auto* Sum = Builder.CreateExtractElement(Vec, (uint64_t)0);
    for (unsigned i = 1; i < Width; ++i) {
        auto* Elem = Builder.CreateExtractElement(Vec, i);
        Sum = Builder.CreateFAdd(Sum, Elem);
    }
    return Sum;
}

void SPIRVGenerator::addMemoryAttributes(llvm::Function* Func, unsigned VectorWidth) {
    // Add alignment attributes to pointer arguments
    for (auto& Arg : Func->args()) {
        if (Arg.getType()->isPointerTy()) {
            Arg.addAttr(llvm::Attribute::getWithAlignment(
                Builder.getContext(),
                llvm::Align(VectorWidth * sizeof(float))));
        }
    }
}


bool SPIRVGenerator::generateKernel(clang::ForStmt* Loop,
                                  const VectorizationInfo& Info) {
    if (!Module) {
        initializeModule();
    }

    KernelInfo KInfo;
    KInfo.Name = getKernelName(Loop);
    KInfo.VectorWidth = Info.RecommendedWidth;
    KInfo.IsReduction = Info.IsReduction;
    KInfo.OriginalLoop = Loop;

    // Analyze loop for kernel arguments
    class ArgumentCollector : public clang::RecursiveASTVisitor<ArgumentCollector> {
    public:
        std::vector<std::string>& Args;
        explicit ArgumentCollector(std::vector<std::string>& Args) : Args(Args) {}

        bool VisitDeclRefExpr(clang::DeclRefExpr* Expr) {
            if (auto VD = llvm::dyn_cast<clang::VarDecl>(Expr->getDecl())) {
                if (VD->hasGlobalStorage() || VD->getType()->isPointerType()) {
                    Args.push_back(VD->getNameAsString());
                }
            }
            return true;
        }
    };

    ArgumentCollector Collector(KInfo.Arguments);
    Collector.TraverseStmt(Loop->getBody());

    if (KInfo.IsReduction) {
        return generateReductionKernel(KInfo);
    } else {
        return generateVectorizedLoop(KInfo);
    }
}

void SPIRVGenerator::improveReductionKernel(const KernelInfo& KInfo, llvm::Function* Func) {
    Input = Func->arg_begin();

    // Get global ID
    auto* GlobalId = Builder.CreateCall(getGetGlobalId(), {
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0))
    });

    // Create local memory
    auto* LocalMemTy = llvm::ArrayType::get(FloatTy, KInfo.PreferredWorkGroupSize);
    auto* LocalMem = Builder.CreateAlloca(LocalMemTy, nullptr, "local_mem");

    // Get work-item ID
    auto* LocalId = Builder.CreateCall(getGetLocalId(), {
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0))
    });

    // Load and reduce vector
    auto* Vec = createVectorLoad(Input, KInfo.VectorWidth);
    auto* LocalSum = performVectorReduction(Vec, KInfo.VectorWidth);

    // Store to local memory
    auto* LocalPtr = Builder.CreateInBoundsGEP(FloatTy, LocalMem, {LocalId});
    Builder.CreateStore(LocalSum, LocalPtr);

    // Add barrier
    addBarrier(CLK_LOCAL_MEM_FENCE);

    // Get work-group size
    auto* WGSize = Builder.CreateCall(getGetLocalSize(), {
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0))
    });

    // Create work-group reduction
    createWorkGroupReduction(LocalMem, WGSize, LocalId, KInfo);

    // Only leader thread performs atomic update
    auto* IsLeader = Builder.CreateICmpEQ(LocalId,
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0)));

    auto* AtomicBlock = llvm::BasicBlock::Create(Builder.getContext(), "atomic", Func);
    auto* ExitBlock = llvm::BasicBlock::Create(Builder.getContext(), "exit", Func);

    Builder.CreateCondBr(IsLeader, AtomicBlock, ExitBlock);

    // Generate atomic update code
    Builder.SetInsertPoint(AtomicBlock);

    auto* Result = std::next(Func->arg_begin());
    auto* FinalSum = Builder.CreateLoad(FloatTy, LocalMem);

    Builder.CreateAtomicRMW(
        llvm::AtomicRMWInst::FAdd,
        Result,
        FinalSum,
        llvm::MaybeAlign(4),
        llvm::AtomicOrdering::SequentiallyConsistent
    );

    Builder.CreateBr(ExitBlock);

    // Set insertion point to exit block
    Builder.SetInsertPoint(ExitBlock);
}

void SPIRVGenerator::addSPIRVMetadata(llvm::Function* Func) {
    // Add SPIR-V calling convention
    Func->setCallingConv(llvm::CallingConv::SPIR_KERNEL);

    // Add SPIR-V memory model metadata
    llvm::Module* M = Func->getParent();
    // Create constant metadata values
    auto* SourceVal = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Builder.getContext()), 0);
    auto* VersionVal = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Builder.getContext()), 100);
    auto* MemModelVal = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Builder.getContext()), 1);

    M->addModuleFlag(llvm::Module::Warning, "spirv.Source",
                     llvm::ConstantAsMetadata::get(SourceVal));
    M->addModuleFlag(llvm::Module::Warning, "spirv.SourceVersion",
                     llvm::ConstantAsMetadata::get(VersionVal));
    M->addModuleFlag(llvm::Module::Warning, "spirv.MemoryModel",
                     llvm::ConstantAsMetadata::get(MemModelVal));

    // Add kernel attribute
    Func->addFnAttr("opencl.kernels", Func->getName());
}

void SPIRVGenerator::createWorkGroupReduction(
    llvm::Value* LocalMem,
    llvm::Value* WGSize,
    llvm::Value* LocalId,
    const KernelInfo& KInfo) {

    auto* Func = Builder.GetInsertBlock()->getParent();

    // Create reduction loop blocks
    auto* ReduceEntry = llvm::BasicBlock::Create(Builder.getContext(), "reduce_entry", Func);
    Builder.CreateBr(ReduceEntry);
    Builder.SetInsertPoint(ReduceEntry);

    for (unsigned StepSize = 1; StepSize < KInfo.PreferredWorkGroupSize; StepSize *= 2) {
        auto* ReduceBlock = llvm::BasicBlock::Create(
            Builder.getContext(), "reduce_" + std::to_string(StepSize), Func);
        auto* ContinueBlock = llvm::BasicBlock::Create(
            Builder.getContext(), "continue_" + std::to_string(StepSize), Func);

        // Create condition for this reduction step
        auto* StepVal = llvm::ConstantInt::get(
            llvm::Type::getInt32Ty(Builder.getContext()), StepSize);
        auto* InRange = Builder.CreateICmpULT(
            Builder.CreateAdd(LocalId, StepVal),
            WGSize
        );

        Builder.CreateCondBr(InRange, ReduceBlock, ContinueBlock);

        // Generate reduction code
        Builder.SetInsertPoint(ReduceBlock);
        auto* Ptr1 = Builder.CreateInBoundsGEP(FloatTy, LocalMem, {LocalId});
        auto* Ptr2 = Builder.CreateInBoundsGEP(FloatTy, LocalMem,
            {Builder.CreateAdd(LocalId, StepVal)});

        auto* Val1 = Builder.CreateLoad(FloatTy, Ptr1);
        auto* Val2 = Builder.CreateLoad(FloatTy, Ptr2);
        auto* Sum = Builder.CreateFAdd(Val1, Val2);
        Builder.CreateStore(Sum, Ptr1);
        Builder.CreateBr(ContinueBlock);

        // Continue with next iteration
        Builder.SetInsertPoint(ContinueBlock);
        addBarrier(CLK_LOCAL_MEM_FENCE);
    }
}

void SPIRVGenerator::addWorkGroupSizeHint(llvm::Function* Func, unsigned Size) {
    llvm::Metadata* MDArgs[] = {
        llvm::MDString::get(Builder.getContext(), "reqd_work_group_size"),
        llvm::ConstantAsMetadata::get(
            llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, Size)))
    };
    Func->addMetadata("opencl.kernels",
        *llvm::MDNode::get(Builder.getContext(), MDArgs));
}

void SPIRVGenerator::improveSimpleVectorization(const KernelInfo& KInfo, llvm::Function* Func) {
    // Initialize FloatTy if not already done
    FloatTy = llvm::Type::getFloatTy(Builder.getContext());

    // Create entry block
    auto* Entry = llvm::BasicBlock::Create(Builder.getContext(), "entry", Func);
    Builder.SetInsertPoint(Entry);

    // Get global ID
    auto* GlobalId = Builder.CreateCall(getGetGlobalId(), {
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0))
    });

    // Ensure we have valid arguments
    if (Func->arg_size() < 3) {
        llvm::errs() << "Error: Function requires at least 3 arguments\n";
        Builder.CreateRetVoid();
        return;
    }

    auto* Input = Func->arg_begin();
    auto* Output = std::next(Func->arg_begin());
    auto* N = std::next(Func->arg_begin(), 2);  // Global size argument

    // Create vector and scalar blocks
    auto* VectorBlock = llvm::BasicBlock::Create(Builder.getContext(), "vector", Func);
    auto* ScalarBlock = llvm::BasicBlock::Create(Builder.getContext(), "scalar", Func);
    auto* ExitBlock = llvm::BasicBlock::Create(Builder.getContext(), "exit", Func);

    // Create branch condition
    auto* VecCheck = Builder.CreateICmpULT(
        Builder.CreateAdd(GlobalId,
            llvm::ConstantInt::get(Builder.getContext(),
                llvm::APInt(32, KInfo.VectorWidth - 1))),
        N
    );

    // Branch from entry to vector/scalar blocks
    Builder.CreateCondBr(VecCheck, VectorBlock, ScalarBlock);

    // Set up vector block
    Builder.SetInsertPoint(VectorBlock);
    auto* VecPtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Input,
        {GlobalId},
        "vec_load_ptr"
    );
    auto* Vec = createVectorLoad(VecPtr, KInfo.VectorWidth);

    // Apply vector operation
    auto* Result = Vec;  // This will be modified based on the operation type

    // Create vector store pointer
    auto* VecStorePtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Output,
        {GlobalId},
        "vec_store_ptr"
    );
    createVectorStore(Result, VecStorePtr);
    Builder.CreateBr(ExitBlock);

    // Set up scalar block
    Builder.SetInsertPoint(ScalarBlock);
    auto* ScalarLoadPtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Input,
        {GlobalId},
        "scalar_load_ptr"
    );
    auto* ScalarVal = Builder.CreateLoad(FloatTy, ScalarLoadPtr);

    auto* ScalarStorePtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Output,
        {GlobalId},
        "scalar_store_ptr"
    );
    Builder.CreateStore(ScalarVal, ScalarStorePtr);
    Builder.CreateBr(ExitBlock);

    // Set up exit block
    Builder.SetInsertPoint(ExitBlock);
    Builder.CreateRetVoid();

    // Add attributes and metadata
    addMemoryAttributes(Func, KInfo.VectorWidth);
    addWorkGroupSizeHint(Func, KInfo.PreferredWorkGroupSize);
}

bool SPIRVGenerator::generateVectorizedLoop(const KernelInfo& KInfo) {
    // Initialize FloatTy if not already done
    FloatTy = llvm::Type::getFloatTy(Builder.getContext());

    // Create kernel function type
    std::vector<llvm::Type*> ArgTypes;
    for (const auto& _ : KInfo.Arguments) {
        ArgTypes.push_back(llvm::PointerType::get(FloatTy, 0));
    }
    ArgTypes.push_back(llvm::Type::getInt32Ty(Builder.getContext()));

    auto* FuncTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(Builder.getContext()),
        ArgTypes,
        false
    );

    // Create kernel function
    auto* Func = llvm::Function::Create(
        FuncTy,
        llvm::Function::ExternalLinkage,
        KInfo.Name,
        Module.get()
    );

    // Add OpenCL kernel attribute
    Func->addFnAttr("opencl.kernels", KInfo.Name);

    // Create entry block
    auto* Entry = llvm::BasicBlock::Create(Builder.getContext(), "entry", Func);
    Builder.SetInsertPoint(Entry);

    // Get global ID
    auto* GlobalId = Builder.CreateCall(getGetGlobalId(), {
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0))
    });

    auto* Input = Func->arg_begin();
    auto* Output = std::next(Func->arg_begin());
    auto* N = std::next(Func->arg_begin(), 2);

    // Create vector and scalar blocks
    auto* VectorBlock = llvm::BasicBlock::Create(Builder.getContext(), "vector", Func);
    auto* ScalarBlock = llvm::BasicBlock::Create(Builder.getContext(), "scalar", Func);
    auto* ExitBlock = llvm::BasicBlock::Create(Builder.getContext(), "exit", Func);

    // Create branch condition
    auto* VecCheck = Builder.CreateICmpULT(
        Builder.CreateAdd(GlobalId,
            llvm::ConstantInt::get(Builder.getContext(),
                llvm::APInt(32, KInfo.VectorWidth - 1))),
        N
    );

    Builder.CreateCondBr(VecCheck, VectorBlock, ScalarBlock);

    // Set up vector block
    Builder.SetInsertPoint(VectorBlock);
    auto* VecPtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Input,
        {GlobalId},
        "vec_load_ptr"
    );

    // Load vector
    auto* Vec = createVectorLoad(VecPtr, KInfo.VectorWidth);

    // Analyze and generate vector operation
    class OperationAnalyzer : public clang::RecursiveASTVisitor<OperationAnalyzer> {
    public:
        enum OpType { None, Add, Mul, Sub, Div };
        OpType Operation = None;
        llvm::APFloat Constant = llvm::APFloat(0.0f);
        bool HasOperation = false;

        bool VisitBinaryOperator(clang::BinaryOperator* BO) {
            if (BO->isAssignmentOp()) {
                if (auto* RHS = llvm::dyn_cast<clang::BinaryOperator>(
                        BO->getRHS()->IgnoreParenImpCasts())) {
                    switch (RHS->getOpcode()) {
                        case clang::BO_Add: Operation = Add; break;
                        case clang::BO_Mul: Operation = Mul; break;
                        case clang::BO_Sub: Operation = Sub; break;
                        case clang::BO_Div: Operation = Div; break;
                        default: break;
                    }

                    if (auto* FL = llvm::dyn_cast<clang::FloatingLiteral>(
                            RHS->getRHS()->IgnoreParenImpCasts())) {
                        Constant = FL->getValue();
                        HasOperation = true;
                    }
                }
            }
            return true;
        }
    };

    OperationAnalyzer OpAnalyzer;
    OpAnalyzer.TraverseStmt(KInfo.OriginalLoop->getBody());

    llvm::Value* Result = Vec;
    if (OpAnalyzer.HasOperation) {
        auto* Constant = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(KInfo.VectorWidth),
            llvm::ConstantFP::get(Builder.getContext(), OpAnalyzer.Constant)
        );

        switch (OpAnalyzer.Operation) {
            case OperationAnalyzer::Add:
                Result = Builder.CreateFAdd(Vec, Constant);
                break;
            case OperationAnalyzer::Mul:
                Result = Builder.CreateFMul(Vec, Constant);
                break;
            case OperationAnalyzer::Sub:
                Result = Builder.CreateFSub(Vec, Constant);
                break;
            case OperationAnalyzer::Div:
                Result = Builder.CreateFDiv(Vec, Constant);
                break;
            default:
                Result = Vec;
        }
    }

    // Store vector result
    auto* VecStorePtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Output,
        {GlobalId},
        "vec_store_ptr"
    );
    createVectorStore(Result, VecStorePtr);
    Builder.CreateBr(ExitBlock);

    // Set up scalar block
    Builder.SetInsertPoint(ScalarBlock);
    auto* ScalarLoadPtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Input,
        {GlobalId},
        "scalar_load_ptr"
    );
    llvm::Value* ScalarVal = Builder.CreateLoad(FloatTy, ScalarLoadPtr);  // Change type to llvm::Value*

    if (OpAnalyzer.HasOperation) {
        auto* Constant = llvm::ConstantFP::get(
            Builder.getContext(),
            OpAnalyzer.Constant
        );
        switch (OpAnalyzer.Operation) {
            case OperationAnalyzer::Add:
                ScalarVal = Builder.CreateFAdd(ScalarVal, Constant);
                break;
            case OperationAnalyzer::Mul:
                ScalarVal = Builder.CreateFMul(ScalarVal, Constant);
                break;
            case OperationAnalyzer::Sub:
                ScalarVal = Builder.CreateFSub(ScalarVal, Constant);
                break;
            case OperationAnalyzer::Div:
                ScalarVal = Builder.CreateFDiv(ScalarVal, Constant);
                break;
            default:
                break;
        }
    }

    auto* ScalarStorePtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Output,
        {GlobalId},
        "scalar_store_ptr"
    );
    Builder.CreateStore(ScalarVal, ScalarStorePtr);
    Builder.CreateBr(ExitBlock);

    // Set up exit block
    Builder.SetInsertPoint(ExitBlock);
    Builder.CreateRetVoid();

    // Add attributes and metadata
    addMemoryAttributes(Func, KInfo.VectorWidth);
    addWorkGroupSizeHint(Func, KInfo.PreferredWorkGroupSize);

    return !llvm::verifyFunction(*Func, &llvm::errs());
}


bool SPIRVGenerator::generateReductionKernel(const KernelInfo& KInfo) {
    // Initialize types
    FloatTy = llvm::Type::getFloatTy(Builder.getContext());

    // Create kernel function type
    std::vector<llvm::Type*> ArgTypes;

    // Input array
    ArgTypes.push_back(llvm::PointerType::get(FloatTy, 0));
    // Result buffer
    ArgTypes.push_back(llvm::PointerType::get(FloatTy, 0));
    // Global size
    ArgTypes.push_back(llvm::Type::getInt32Ty(Builder.getContext()));

    auto* FuncTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(Builder.getContext()),
        ArgTypes,
        false
    );

    // Create kernel function
    auto* Func = llvm::Function::Create(
        FuncTy,
        llvm::Function::ExternalLinkage,
        KInfo.Name,
        Module.get()
    );

    // Add attributes
    Func->addFnAttr("opencl.kernels", KInfo.Name);

    // Create entry block first
    auto* Entry = llvm::BasicBlock::Create(Builder.getContext(), "entry", Func);
    Builder.SetInsertPoint(Entry);

    // Now that we have a basic block, improve the reduction kernel
    improveReductionKernel(KInfo, Func);

    // Create return
    Builder.CreateRetVoid();

    // Add memory attributes
    addMemoryAttributes(Func, KInfo.VectorWidth);

    return !llvm::verifyFunction(*Func, &llvm::errs());
}


std::string SPIRVGenerator::getKernelName(clang::ForStmt* Loop) {
    // Generate a unique name based on location
    auto& SM = Context->getSourceManager();
    auto Loc = SM.getSpellingLineNumber(Loop->getBeginLoc());
    return "kernel_line_" + std::to_string(Loc);
}

llvm::Type* SPIRVGenerator::getVectorType(llvm::Type* ElemTy, unsigned Width) {
    return llvm::VectorType::get(ElemTy, Width, false);
}

llvm::Value* SPIRVGenerator::createVectorLoad(llvm::Value* Ptr, unsigned Width) {
    auto* VecTy = llvm::VectorType::get(
        llvm::Type::getFloatTy(Builder.getContext()),
        Width,
        false
    );
    auto* CastPtr = Builder.CreateBitCast(
        Ptr,
        llvm::PointerType::get(VecTy, 0),
        "vecptr_cast"
    );
    return Builder.CreateLoad(VecTy, CastPtr);
}

llvm::Value* SPIRVGenerator::createVectorStore(llvm::Value* Val, llvm::Value* Ptr) {
    auto* VecTy = llvm::cast<llvm::VectorType>(Val->getType());
    auto* CastPtr = Builder.CreateBitCast(
        Ptr,
        llvm::PointerType::get(VecTy, 0),
        "vecptr_cast"
    );
    return Builder.CreateStore(Val, CastPtr);
}

} // namespace cspir
