
#pragma once


#include "types.h"
#include "spirv_generator.h"  // Include this first
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/FileSystem.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Analysis/CFG.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"

namespace cspir {
    class SPIRVGenerator;  // Forward declaration

    class LoopAnalyzer {
    public:
        explicit LoopAnalyzer(clang::ASTContext *Context)
            : Context(Context), Diags(Context->getDiagnostics()) {}

        bool isVectorizable(clang::ForStmt *FS);
        VectorizationInfo analyzeWithOptimizer(clang::ForStmt *FS);

    private:
        bool checkDataAccess(clang::Stmt *Body, VectorizationInfo &Info);
        bool analyzeCFG(clang::ForStmt *FS, VectorizationInfo &Info);
        bool isReductionLoop(clang::ForStmt *FS, VectorizationInfo &Info);
        bool checkTypes(clang::Stmt *Body, VectorizationInfo &Info);
        bool isSimpleVectorizablePattern(clang::ForStmt *FS);  // Add this declaration

        clang::ASTContext *Context;
        clang::DiagnosticsEngine &Diags;
    };


class C89ASTVisitor : public clang::RecursiveASTVisitor<C89ASTVisitor> {
public:
    explicit C89ASTVisitor(clang::ASTContext *Context)
        : Context(Context), loopAnalyzer(Context) {} // Changed to lowercase
    virtual ~C89ASTVisitor() = default;


    bool VisitFunctionDecl(clang::FunctionDecl *FD);
    bool VisitVarDecl(clang::VarDecl *VD);
    bool VisitRecordDecl(clang::RecordDecl *RD);
    bool VisitForStmt(clang::ForStmt *FS);
    bool VisitWhileStmt(clang::WhileStmt *WS);
    bool VisitIfStmt(clang::IfStmt *IS);
    bool VisitCompoundStmt(clang::CompoundStmt *CS);
    bool VisitBinaryOperator(clang::BinaryOperator *BO);
    bool VisitCallExpr(clang::CallExpr *CE);
    bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE);

private:
    clang::ASTContext *Context;
    LoopAnalyzer loopAnalyzer;  // Changed to lowercase
};

class C89ASTConsumer : public clang::ASTConsumer {
public:
    explicit C89ASTConsumer(clang::ASTContext *Context) : Visitor(Context) {}
    virtual ~C89ASTConsumer() override = default;

    void HandleTranslationUnit(clang::ASTContext &Context) override {
        Visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    C89ASTVisitor Visitor;
};

class C89FrontendAction : public clang::ASTFrontendAction {
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
        clang::CompilerInstance &CI, llvm::StringRef /*InFile*/) override {
        return std::make_unique<C89ASTConsumer>(&CI.getASTContext());
    }

    bool BeginSourceFileAction(clang::CompilerInstance & /*CI*/) override {
        return true;
    }
};

class C89Parser {
public:
    C89Parser() = default;
    ~C89Parser() = default;

    bool parseFile(const std::string &FileName);

private:
    void setupToolingArguments(std::vector<std::string>& Args);
};



} // namespace cspir
