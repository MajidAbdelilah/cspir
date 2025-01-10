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
          Builder(*LLVMCtx) {}

    bool generateKernel(clang::ForStmt* Loop, const VectorizationInfo& Info);
    llvm::Module* getModule() { return Module.get(); }



private:
    bool generateVectorizedLoop(const KernelInfo& KInfo);
    bool generateReductionKernel(const KernelInfo& KInfo);
    llvm::Value* createVectorLoad(llvm::Value* Ptr, unsigned Width);
    llvm::Value* createVectorStore(llvm::Value* Val, llvm::Value* Ptr);

    clang::ASTContext* Context;
    std::unique_ptr<llvm::LLVMContext> LLVMCtx;
    llvm::IRBuilder<> Builder;
    std::unique_ptr<llvm::Module> Module;

    std::string getKernelName(clang::ForStmt* Loop);
    llvm::Type* getVectorType(llvm::Type* ElemTy, unsigned Width);
    void initializeModule();
};

} // namespace cspir
