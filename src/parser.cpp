// Parser.cpp
#include "parser.h"
#include "clang/Tooling/CompilationDatabase.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Tooling/CommonOptionsParser.h"

namespace cspir {

    bool LoopAnalyzer::isSimpleVectorizablePattern(clang::ForStmt *FS) {
        class PatternMatcher : public clang::RecursiveASTVisitor<PatternMatcher> {
        public:
            bool IsSimplePattern = false;
            bool HasDependency = false;

            bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE) {
                // Check for array index expressions
                if (auto *Idx = ASE->getIdx()->IgnoreParenImpCasts()) {
                    // Check for array[i-1] pattern
                    if (auto *BO = llvm::dyn_cast<clang::BinaryOperator>(Idx)) {
                        if (BO->getOpcode() == clang::BO_Sub) {
                            // If we find a subtraction in the index, it might be a dependency
                            if (auto *RHS = llvm::dyn_cast<clang::IntegerLiteral>(
                                    BO->getRHS()->IgnoreParenImpCasts())) {
                                if (RHS->getValue().getLimitedValue() == 1) {
                                    HasDependency = true;
                                }
                            }
                        }
                    }
                }
                return true;
            }

            bool VisitBinaryOperator(clang::BinaryOperator *BO) {
                if (BO->getOpcode() == clang::BO_Assign) {
                    if (llvm::isa<clang::ArraySubscriptExpr>(
                            BO->getLHS()->IgnoreParenImpCasts())) {
                        // Check RHS for simple operation
                        if (auto *BinOp = llvm::dyn_cast<clang::BinaryOperator>(
                                BO->getRHS()->IgnoreParenImpCasts())) {
                            if (isSimpleOperation(BinOp) && !HasDependency) {
                                IsSimplePattern = true;
                            }
                        }
                    }
                }
                return true;
            }

        private:
            bool isSimpleOperation(clang::BinaryOperator *BO) {
                if (BO->isMultiplicativeOp() || BO->isAdditiveOp()) {
                    auto *LHS = BO->getLHS()->IgnoreParenImpCasts();
                    auto *RHS = BO->getRHS()->IgnoreParenImpCasts();

                    bool HasArrayAccess = false;
                    bool HasConstant = false;

                    if (llvm::isa<clang::ArraySubscriptExpr>(LHS)) {
                        HasArrayAccess = true;
                    }
                    if (llvm::isa<clang::FloatingLiteral>(RHS) ||
                        llvm::isa<clang::IntegerLiteral>(RHS)) {
                        HasConstant = true;
                    }

                    return HasArrayAccess && HasConstant;
                }
                return false;
            }
        };

        PatternMatcher Matcher;
        Matcher.TraverseStmt(FS->getBody());
        return Matcher.IsSimplePattern && !Matcher.HasDependency;
    }

    bool LoopAnalyzer::checkTypes(clang::Stmt *Body, VectorizationInfo &Info) {
        class TypeChecker : public clang::RecursiveASTVisitor<TypeChecker> {
        public:
            bool HasMixedTypes = false;
            std::vector<std::string> &Reasons;
            llvm::SmallSet<clang::QualType, 4> ComputationTypes;
            llvm::SmallSet<clang::QualType, 4> IndexTypes;

            explicit TypeChecker(std::vector<std::string> &R) : Reasons(R) {}

            bool VisitExpr(clang::Expr *E) {
                auto Type = E->getType();
                if (!Type.isNull()) {
                    if (auto *ASE = llvm::dyn_cast<clang::ArraySubscriptExpr>(E)) {
                        // Array element type
                        Type = ASE->getType();
                        if (Type->isFloatingType() || Type->isIntegerType()) {
                            ComputationTypes.insert(Type);
                        }
                        // Index type should be ignored for mixed type check
                        IndexTypes.insert(ASE->getIdx()->getType());
                    } else if (auto *BO = llvm::dyn_cast<clang::BinaryOperator>(E)) {
                        if (isComputationOp(BO)) {
                            Type = BO->getType();
                            if (Type->isFloatingType() || Type->isIntegerType()) {
                                if (!IndexTypes.count(Type)) {  // Ignore index types
                                    ComputationTypes.insert(Type);
                                }
                            }
                        }
                    }

                    if (ComputationTypes.size() > 1) {
                        HasMixedTypes = true;
                        Reasons.push_back("Mixed computation types detected in loop");
                    }
                }

                return true;
            }

        private:
            bool isComputationOp(clang::BinaryOperator *BO) {
                return BO->getOpcode() == clang::BO_Mul ||
                       BO->getOpcode() == clang::BO_Div ||
                       BO->getOpcode() == clang::BO_Add ||
                       BO->getOpcode() == clang::BO_Sub ||
                       BO->getOpcode() == clang::BO_AddAssign ||
                       BO->getOpcode() == clang::BO_SubAssign ||
                       BO->getOpcode() == clang::BO_MulAssign ||
                       BO->getOpcode() == clang::BO_DivAssign;
            }
        };

        TypeChecker Checker(Info.Reasons);
        Checker.TraverseStmt(Body);
        return !Checker.HasMixedTypes;
    }



    bool LoopAnalyzer::isReductionLoop(clang::ForStmt *FS, VectorizationInfo &Info) {
        class ReductionChecker : public clang::RecursiveASTVisitor<ReductionChecker> {
        public:
            bool IsReduction = false;
            std::string ReductionVar;
            std::vector<std::string> &Reasons;

            explicit ReductionChecker(std::vector<std::string> &R) : Reasons(R) {}

            bool VisitBinaryOperator(clang::BinaryOperator *BO) {
                if (BO->isCompoundAssignmentOp()) {
                    if (auto *LHS = llvm::dyn_cast<clang::DeclRefExpr>(
                            BO->getLHS()->IgnoreParenImpCasts())) {
                        ReductionVar = LHS->getDecl()->getNameAsString();
                        IsReduction = true;
                        Reasons.push_back("Reduction operation detected on variable: " + ReductionVar);
                    }
                }
                return true;
            }
        };

        ReductionChecker Checker(Info.Reasons);
        Checker.TraverseStmt(FS->getBody());
        return Checker.IsReduction;
    }

    VectorizationInfo LoopAnalyzer::analyzeWithOptimizer(clang::ForStmt *FS) {
        VectorizationInfo Info{
            .IsVectorizable = false,
            .Reasons = {},
            .RecommendedWidth = 0,
            .IsReduction = false,
            .IsSimplePattern = false,
            .HasConstantTripCount = false,
            .TripCount = 0
        };

        // First check for dependencies
        class DependencyChecker : public clang::RecursiveASTVisitor<DependencyChecker> {
        public:
            bool HasDependencies = false;
            std::vector<std::string>& Reasons;

            explicit DependencyChecker(std::vector<std::string>& R) : Reasons(R) {}

            bool VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE) {
                if (auto *Idx = ASE->getIdx()->IgnoreParenImpCasts()) {
                    if (auto *BO = llvm::dyn_cast<clang::BinaryOperator>(Idx)) {
                        if (BO->getOpcode() == clang::BO_Sub) {
                            if (auto *RHS = llvm::dyn_cast<clang::IntegerLiteral>(
                                    BO->getRHS()->IgnoreParenImpCasts())) {
                                if (RHS->getValue().getLimitedValue() == 1) {
                                    HasDependencies = true;
                                    Reasons.push_back("Loop-carried dependency detected: array[i-1] access pattern");
                                }
                            }
                        }
                    }
                }
                return true;
            }
        };

        DependencyChecker DepChecker(Info.Reasons);
        DepChecker.TraverseStmt(FS->getBody());
        bool HasDependencies = DepChecker.HasDependencies;

        // Rest of your existing analysis...
        // Check trip count
        if (auto *Cond = llvm::dyn_cast<clang::BinaryOperator>(FS->getCond())) {
            if (auto *RHS = llvm::dyn_cast<clang::IntegerLiteral>(Cond->getRHS())) {
                Info.TripCount = RHS->getValue().getLimitedValue();
                Info.HasConstantTripCount = true;
                Info.Reasons.push_back("Loop trip count: " + std::to_string(Info.TripCount));
            }
        }

        // Check if it's a simple pattern
        Info.IsSimplePattern = isSimpleVectorizablePattern(FS);
        if (Info.IsSimplePattern) {
            Info.Reasons.push_back("Simple vectorizable pattern detected");
        }

        // Check for reduction pattern
        Info.IsReduction = isReductionLoop(FS, Info);

        // Make vectorization decision
        Info.IsVectorizable = (Info.HasConstantTripCount || Info.IsReduction || Info.IsSimplePattern) &&
                             (!HasDependencies || Info.IsReduction) &&  // Changed this line
                             checkTypes(FS->getBody(), Info);

        if (Info.IsVectorizable) {
            if (Info.IsReduction) {
                Info.RecommendedWidth = 4;
            } else if (Info.HasConstantTripCount && Info.TripCount >= 8) {
                Info.RecommendedWidth = 8;
            } else {
                Info.RecommendedWidth = 4;
            }
        } else if (HasDependencies) {  // Add this condition
            Info.Reasons.push_back("Loop cannot be vectorized due to dependencies");
        }

        return Info;
    }


    bool LoopAnalyzer::isVectorizable(clang::ForStmt *FS) {
        auto Info = analyzeWithOptimizer(FS);

        // Print basic analysis header
        llvm::outs() << "\nLLVM Vectorization Analysis:\n";
        llvm::outs() << "-------------------------\n";

        // Print loop location
        llvm::outs() << "Location: ";
        FS->getBeginLoc().print(llvm::outs(), Context->getSourceManager());
        llvm::outs() << "\n\n";

        // Print analysis reasons
        for (const auto& Reason : Info.Reasons) {
            llvm::outs() << "- " << Reason << "\n";
        }

        if (Info.IsVectorizable) {
            // Print detailed analysis for vectorizable loops
            llvm::outs() << "\nVectorization Analysis Details:\n";
            llvm::outs() << "- Pattern: "
                         << (Info.IsReduction ? "Reduction" :
                            Info.IsSimplePattern ? "Simple arithmetic" : "General parallel") << "\n";
            llvm::outs() << "- Vector width: " << Info.RecommendedWidth << "\n";
            llvm::outs() << "- Trip count: "
                         << (Info.HasConstantTripCount ? std::to_string(Info.TripCount) : "Variable") << "\n";

            // Add kernel generation
            SPIRVGenerator Generator(Context);
            if (Generator.generateKernel(FS, Info)) {
                llvm::outs() << "\nGenerated SPIR-V kernel:\n";
                llvm::outs() << "-------------------------\n";
                Generator.getModule()->print(llvm::outs(), nullptr);
            } else {
                llvm::outs() << "\nFailed to generate SPIR-V kernel\n";
            }
        } else {
            llvm::outs() << "\nLoop is not vectorizable\n";
        }

        return Info.IsVectorizable;
    }

    bool C89ASTVisitor::VisitRecordDecl(clang::RecordDecl *RD) {
        llvm::outs() << "\nRecord Declaration: (";
        RD->getLocation().print(llvm::outs(), Context->getSourceManager());
        llvm::outs() << ")\n";
        llvm::outs() << "  Name: " << RD->getNameAsString() << "\n";
        llvm::outs() << "  Kind: " << (RD->isStruct() ? "struct" :
                                      RD->isUnion() ? "union" : "record") << "\n";
        llvm::outs() << "  Size: " << Context->getTypeSize(RD->getTypeForDecl()) << " bits\n";
        llvm::outs() << "  Alignment: " << Context->getTypeAlign(RD->getTypeForDecl()) << " bits\n";

        // Print fields
        llvm::outs() << "  Fields:\n";
        for (const auto *Field : RD->fields()) {
            llvm::outs() << "    - " << Field->getNameAsString() << ": ";
            Field->getType().print(llvm::outs(), Context->getPrintingPolicy());
            llvm::outs() << " (offset: " << Context->getFieldOffset(Field) << " bits)\n";
        }

        return true;
    }

    bool C89ASTVisitor::VisitForStmt(clang::ForStmt *FS) {
        llvm::outs() << "\nFor Loop:\n";

        // Print loop structure
        if (auto *InitStmt = FS->getInit()) {
            llvm::outs() << "  Init: ";
            InitStmt->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
            llvm::outs() << "\n";
        }

        if (auto *Cond = FS->getCond()) {
            llvm::outs() << "  Condition: ";
            Cond->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
            llvm::outs() << "\n";
        }

        if (auto *Inc = FS->getInc()) {
            llvm::outs() << "  Increment: ";
            Inc->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
            llvm::outs() << "\n";
        }

        // Add vectorization analysis
        llvm::outs() << "\nAnalyzing loop for vectorization:\n";
        /*bool IsVectorizable =*/ loopAnalyzer.isVectorizable(FS);  // Remove unused variable

        return true;
    }

    bool C89ASTVisitor::VisitWhileStmt(clang::WhileStmt *WS) {
        llvm::outs() << "\nWhile Loop:\n";
        llvm::outs() << "  Condition: ";
        WS->getCond()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
        llvm::outs() << "\n";

        return true;
    }

    bool C89ASTVisitor::VisitFunctionDecl(clang::FunctionDecl *FD) {
        llvm::outs() << "\nFunction Declaration: (";
        FD->getLocation().print(llvm::outs(), Context->getSourceManager());
        llvm::outs() << ")\n";
        llvm::outs() << "  Name: " << FD->getNameAsString() << "\n";
        llvm::outs() << "  Return Type: ";
        FD->getReturnType().print(llvm::outs(), Context->getPrintingPolicy());
        llvm::outs() << "\n";
        llvm::outs() << "  Storage Class: " << FD->getStorageClass() << "\n";

        // Print parameters
        llvm::outs() << "  Parameters:\n";
        for (auto *Param : FD->parameters()) {
            llvm::outs() << "    - " << Param->getNameAsString() << ": ";
            Param->getType().print(llvm::outs(), Context->getPrintingPolicy());
            llvm::outs() << "\n";
        }

        return true;
    }


    bool C89ASTVisitor::VisitVarDecl(clang::VarDecl *VD) {
        llvm::outs() << "\nVariable Declaration: (";
        VD->getLocation().print(llvm::outs(), Context->getSourceManager());
        llvm::outs() << ")\n";
        llvm::outs() << "  Name: " << VD->getNameAsString() << "\n";
        llvm::outs() << "  Type: ";
        VD->getType().print(llvm::outs(), Context->getPrintingPolicy());
        llvm::outs() << "\n";
        llvm::outs() << "  Storage Class: " << VD->getStorageClass() << "\n";
        llvm::outs() << "  Scope: " << (VD->isFileVarDecl() ? "file" :
                                       VD->hasGlobalStorage() ? "global" : "local") << "\n";

        if (VD->hasInit()) {
            llvm::outs() << "  Initializer: ";
            VD->getInit()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
            llvm::outs() << "\n";
        }

        return true;
    }

    bool C89ASTVisitor::VisitIfStmt(clang::IfStmt *IS) {
        llvm::outs() << "\nIf Statement:\n";
        llvm::outs() << "  Condition: ";
        IS->getCond()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
        llvm::outs() << "\n";

        return true;
    }

    bool C89ASTVisitor::VisitBinaryOperator(clang::BinaryOperator *BO) {
        llvm::outs() << "\nBinary Operation: (";
        BO->getOperatorLoc().print(llvm::outs(), Context->getSourceManager());
        llvm::outs() << ")\n";
        llvm::outs() << "  Operator: " << BO->getOpcodeStr().str() << "\n";
        llvm::outs() << "  Result Type: ";
        BO->getType().print(llvm::outs(), Context->getPrintingPolicy());
        llvm::outs() << "\n";
        llvm::outs() << "  Left: ";
        BO->getLHS()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
        llvm::outs() << " (Type: ";
        BO->getLHS()->getType().print(llvm::outs(), Context->getPrintingPolicy());
        llvm::outs() << ")\n";
        llvm::outs() << "  Right: ";
        BO->getRHS()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
        llvm::outs() << " (Type: ";
        BO->getRHS()->getType().print(llvm::outs(), Context->getPrintingPolicy());
        llvm::outs() << ")\n";

        return true;
    }

    bool C89ASTVisitor::VisitCompoundStmt(clang::CompoundStmt *CS) {
        llvm::outs() << "\nCompound Statement (Block):\n";
        llvm::outs() << "  Number of statements: " << CS->size() << "\n";
        return true;
    }

    bool C89ASTVisitor::VisitCallExpr(clang::CallExpr *CE) {
        llvm::outs() << "\nFunction Call:\n";
        if (auto *FD = CE->getDirectCallee()) {
            llvm::outs() << "  Function: " << FD->getNameAsString() << "\n";
        }

        llvm::outs() << "  Arguments:\n";
        for (unsigned i = 0; i < CE->getNumArgs(); ++i) {
            llvm::outs() << "    " << i << ": ";
            CE->getArg(i)->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
            llvm::outs() << "\n";
        }

        return true;
    }

    bool C89ASTVisitor::VisitArraySubscriptExpr(clang::ArraySubscriptExpr *ASE) {
        llvm::outs() << "\nArray Subscript:\n";
        llvm::outs() << "  Base: ";
        ASE->getBase()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
        llvm::outs() << "\n  Index: ";
        ASE->getIdx()->printPretty(llvm::outs(), nullptr, Context->getPrintingPolicy());
        llvm::outs() << "\n";

        return true;
    }

bool C89Parser::parseFile(const std::string &FileName) {
    // Get absolute path of the input file
    llvm::SmallString<256> AbsolutePath;
    std::error_code EC = llvm::sys::fs::real_path(FileName, AbsolutePath);
    if (EC) {
        llvm::errs() << "Error: Could not get real path for " << FileName << ": " << EC.message() << "\n";
        return false;
    }

    // Get the directory containing the input file
    llvm::StringRef Directory = llvm::sys::path::parent_path(AbsolutePath);
    if (Directory.empty()) {
        // Get current directory
        llvm::SmallString<256> CurrentDir;
        EC = llvm::sys::fs::current_path(CurrentDir);
        if (EC) {
            llvm::errs() << "Error: Could not get current directory: " << EC.message() << "\n";
            return false;
        }
        Directory = CurrentDir.str();
    }

    // Set up the compilation arguments
    std::vector<std::string> CommandLine;
    setupToolingArguments(CommandLine);

    // Add the input file to the command line
    CommandLine.push_back(AbsolutePath.str().str());

    // Create compilation database
    std::vector<clang::tooling::CompileCommand> Commands;
    Commands.push_back(clang::tooling::CompileCommand(
        Directory.str(),                                // Directory
        llvm::sys::path::filename(AbsolutePath).str(), // Filename
        CommandLine,                                   // Command line arguments
        ""                                            // Output file
    ));

    class CompilationDatabase : public clang::tooling::CompilationDatabase {
    public:
        CompilationDatabase(std::vector<clang::tooling::CompileCommand> Commands)
            : Commands(std::move(Commands)) {}

        std::vector<clang::tooling::CompileCommand>
        getCompileCommands(llvm::StringRef /*FilePath*/) const override {
            return Commands;
        }

        std::vector<clang::tooling::CompileCommand>
        getAllCompileCommands() const override {
            return Commands;
        }

    private:
        std::vector<clang::tooling::CompileCommand> Commands;
    };

    // Create the compilation database
    auto DB = std::make_unique<CompilationDatabase>(std::move(Commands));

    // Create ClangTool
    std::vector<std::string> Sources;
    Sources.push_back(AbsolutePath.str().str());

    clang::tooling::ClangTool Tool(*DB, Sources);

    // Add resource directory
    Tool.appendArgumentsAdjuster(
        [](const clang::tooling::CommandLineArguments &Args, llvm::StringRef) {
            std::vector<std::string> NewArgs = Args;
            NewArgs.push_back("-resource-dir");
            NewArgs.push_back("\"" CLANG_RESOURCE_DIR "\"");  // Add quotes
            return NewArgs;
        }
    );

    // Run the tool with our frontend action
    return !Tool.run(clang::tooling::newFrontendActionFactory<C89FrontendAction>().get());
}
void C89Parser::setupToolingArguments(std::vector<std::string>& Args) {
    // Add compiler name as first argument
    Args.push_back("clang");

    // Add optimization and vectorization flags
    Args.push_back("-O3");
    Args.push_back("-fvectorize");
    Args.push_back("-fslp-vectorize");
    Args.push_back("-march=native");
    Args.push_back("-ffast-math");

    // Add basic C compilation flags
    Args.push_back("-x");
    Args.push_back("c");
    Args.push_back("-std=c89");
    Args.push_back("-pedantic");
    Args.push_back("-fno-gnu-keywords");

    // Add debug info for better analysis
    Args.push_back("-g");

    // Add system includes
    Args.push_back("-I/usr/include");
    Args.push_back("-I/usr/local/include");

    // Add target
    Args.push_back("-target");
    Args.push_back("x86_64-unknown-linux-gnu");
}
} // namespace cspir
