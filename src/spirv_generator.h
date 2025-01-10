#pragma once

#include "types.h"
#include "clang/AST/ASTContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Constants.h"
#include <memory>

namespace cspir {
    class SPIRVGenerator {
    public:
        explicit SPIRVGenerator(clang::ASTContext* Context)
            : Context(Context),
                LLVMCtx(std::make_unique<llvm::LLVMContext>()),
                FloatTy(nullptr),
                Input(nullptr),
                Builder(*LLVMCtx)  // Move Builder initialization to match declaration order
        {
            initializeModule();
        }

        bool generateKernel(clang::ForStmt* Loop, const VectorizationInfo& Info);
        llvm::Module* getModule() { return Module.get(); }
    private:
    // Add member variables for commonly used types
       llvm::Type* FloatTy = nullptr;
       llvm::Value* Input = nullptr;  // Add Input member variable

       // OpenCL function declarations
       llvm::FunctionCallee getGetGlobalId() {
           return getOpenCLFunction(OpenCLBuiltins::GET_GLOBAL_ID,
               llvm::Type::getInt32Ty(Builder.getContext()),
               {llvm::Type::getInt32Ty(Builder.getContext())});
       }

        llvm::FunctionCallee getGetLocalId() {
            return getOpenCLFunction(OpenCLBuiltins::GET_LOCAL_ID,
                llvm::Type::getInt32Ty(Builder.getContext()),
                {llvm::Type::getInt32Ty(Builder.getContext())});
        }

        llvm::FunctionCallee getGetGroupId() {
            return getOpenCLFunction(OpenCLBuiltins::GET_GROUP_ID,
                llvm::Type::getInt32Ty(Builder.getContext()),
                {llvm::Type::getInt32Ty(Builder.getContext())});
        }

        llvm::FunctionCallee getGetLocalSize() {
            return getOpenCLFunction(OpenCLBuiltins::GET_LOCAL_SIZE,
                llvm::Type::getInt32Ty(Builder.getContext()),
                {llvm::Type::getInt32Ty(Builder.getContext())});
        }

        // Kernel generation helpers
        void addBarrier();
        void addBarrier(unsigned Fence);  // Add overload for fence type
        void addMemoryAttributes(llvm::Function* Func, unsigned VectorWidth);

        void addSPIRVMetadata(llvm::Function* Func);

        void addWorkGroupSizeHint(llvm::Function* Func, unsigned Size);
        // Helper functions for metadata
        void addKernelMetadata(llvm::Function* Func);
        void addWorkGroupMetadata(llvm::Function* Func, unsigned Size);
        void addMemoryModelMetadata(llvm::Module* M);

        // Main kernel generation functions
        bool generateVectorizedLoop(const KernelInfo& KInfo);
        bool generateReductionKernel(const KernelInfo& KInfo);

        // Vector operation helpers
        llvm::Value* createVectorLoad(llvm::Value* Ptr, unsigned Width);
        llvm::Value* createVectorStore(llvm::Value* Val, llvm::Value* Ptr);
        llvm::Value* performVectorReduction(llvm::Value* Vec, unsigned Width);

        // Optimization helpers
        void improveSimpleVectorization(const KernelInfo& KInfo, llvm::Function* Func);
        void improveReductionKernel(const KernelInfo& KInfo, llvm::Function* Func);
        // Update createWorkGroupReduction declaration to include KInfo
            void createWorkGroupReduction(
                llvm::Value* LocalMem,
                llvm::Value* WGSize,
                llvm::Value* LocalId,
                const KernelInfo& KInfo);

        // OpenCL function declarations
        llvm::FunctionCallee getOpenCLFunction(const std::string& Name,
                                                llvm::Type* RetTy,
                                                llvm::ArrayRef<llvm::Type*> ArgTypes);

        // Utility functions
        std::string getKernelName(clang::ForStmt* Loop);
        llvm::Type* getVectorType(llvm::Type* ElemTy, unsigned Width);
        void initializeModule();

        // Class members
        clang::ASTContext* Context;
        std::unique_ptr<llvm::LLVMContext> LLVMCtx;
        llvm::IRBuilder<> Builder;
        std::unique_ptr<llvm::Module> Module;
    };
} // namespace cspir
