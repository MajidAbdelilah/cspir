#pragma once

#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include <string>
#include <vector>

namespace cspir {
    // OpenCL Memory Fence Constants
    enum OpenCLMemFence {
        CLK_LOCAL_MEM_FENCE  = 1,
        CLK_GLOBAL_MEM_FENCE = 2
    };

    // OpenCL Built-in Functions
    struct OpenCLBuiltins {
        static constexpr const char* GET_GLOBAL_ID = "get_global_id";
        static constexpr const char* GET_LOCAL_ID  = "get_local_id";
        static constexpr const char* GET_GROUP_ID  = "get_group_id";
        static constexpr const char* GET_LOCAL_SIZE = "get_local_size";
        static constexpr const char* BARRIER = "barrier";
    };

// Forward declarations
class LoopAnalyzer;
class SPIRVGenerator;

// Common structures
struct VectorizationInfo {
    bool IsVectorizable;
    std::vector<std::string> Reasons;
    unsigned RecommendedWidth;
    bool IsReduction;
    bool IsSimplePattern;
    bool HasConstantTripCount;
    uint64_t TripCount;
};

struct KernelInfo {
    std::string Name;
    unsigned VectorWidth;
    bool IsReduction;
    std::vector<std::string> Arguments;
    clang::ForStmt* OriginalLoop;
    // Work-group related
    size_t PreferredWorkGroupSize = 256;  // Default size
    size_t MaxWorkGroupSize = 1024;       // Hardware limit
    bool UsesLocalMemory = false;

    // OpenCL specific
    std::vector<std::string> RequiredExtensions;
    std::vector<std::pair<std::string, std::string>> Attributes;
};

} // namespace cspir
