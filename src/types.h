#pragma once

#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include <string>
#include <vector>

namespace cspir {

// Forward declarations
class LoopAnalyzer;
class SPIRVGenerator;

// Common structures
struct VectorizationInfo {
    bool IsVectorizable;
    std::vector<std::string> Reasons;
    unsigned RecommendedWidth;
    bool IsReduction;
};

struct KernelInfo {
    std::string Name;
    unsigned VectorWidth;
    bool IsReduction;
    std::vector<std::string> Arguments;
    clang::ForStmt* OriginalLoop;
};

} // namespace cspir
