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

bool SPIRVGenerator::generateVectorizedLoop(const KernelInfo& KInfo) {
    // Create kernel function type
    std::vector<llvm::Type*> ArgTypes;
    for (const auto& _ : KInfo.Arguments) {
        // For simplicity, assume all arguments are pointer to float
        ArgTypes.push_back(llvm::PointerType::get(
            llvm::Type::getFloatTy(Builder.getContext()),
            0
        ));
    }

    // Add global work size argument
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
    llvm::FunctionCallee GetGlobalId = Module->getOrInsertFunction(
        "get_global_id",
        llvm::Type::getInt32Ty(Builder.getContext()),
        llvm::Type::getInt32Ty(Builder.getContext())
    );
    auto* GlobalId = Builder.CreateCall(GetGlobalId, {
        llvm::ConstantInt::get(Builder.getContext(), llvm::APInt(32, 0))
    });


    // Load vector
    auto* Input = Func->arg_begin();
    auto* FloatTy = llvm::Type::getFloatTy(Builder.getContext());
    auto* PtrTy = llvm::PointerType::get(FloatTy, 0);

    // Create GEP instruction correctly
    auto* VecPtr = Builder.CreateInBoundsGEP(
        FloatTy,
        Input,
        {GlobalId},
        "vecptr"
    );

    // Create vector type and pointer type
    auto* VecTy = llvm::VectorType::get(FloatTy, KInfo.VectorWidth, false);
    auto* VecPtrTy = llvm::PointerType::get(VecTy, 0);

    // Cast scalar pointer to vector pointer
    auto* CastPtr = Builder.CreateBitCast(VecPtr, VecPtrTy, "vec_ptr_cast");

    // Load the vector
    auto* Vec = Builder.CreateLoad(VecTy, CastPtr);

    // Perform vector operation (example: multiply by 2)
    auto* Two = llvm::ConstantVector::getSplat(
        llvm::ElementCount::getFixed(KInfo.VectorWidth),
        llvm::ConstantFP::get(Builder.getContext(), llvm::APFloat(2.0f))
    );
    auto* Result = Builder.CreateFMul(Vec, Two);

    // Store result
    auto* Output = std::next(Func->arg_begin());
    auto* OutPtr = Builder.CreateGEP(
        llvm::Type::getFloatTy(Builder.getContext()),
        Output,
        {GlobalId},
        "outptr"
    );
    createVectorStore(Result, OutPtr);

    // Create return
    Builder.CreateRetVoid();

    // Verify the generated code
    return !llvm::verifyFunction(*Func, &llvm::errs());
}

bool SPIRVGenerator::generateReductionKernel(const KernelInfo& KInfo) {
    // TODO: Implement reduction-specific logic
    llvm::errs() << "Reduction kernel generation not yet implemented for: "
                 << KInfo.Name << "\n";
    return false;
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
