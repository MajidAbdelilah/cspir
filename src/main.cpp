// main.cpp
#include "parser.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        llvm::errs() << "Usage: " << argv[0] << " <source-file>\n";
        return 1;
    }

    cspir::C89Parser Parser;
    if (!Parser.parseFile(argv[1])) {
        llvm::errs() << "Error parsing file\n";
        return 1;
    }

    return 0;
}
