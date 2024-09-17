#include <iostream>
#include <cstdlib> // For std::atoi
#include <cstdio>  // For printf

int main(int argc, char* argv[]) {
    // For command line argument
    if (argc != 2) {
        std::cerr << "Usage: ./task6 N\n";
        return 1;
    }

    // Convert the command line argument to an integer
    int N = std::atoi(argv[1]);

    // 0 to N using printf
    for (int i = 0; i <= N; ++i) {
        std::printf("%d", i);
        if (i < N) {
            std::printf(" "); 
        }
    }
    std::printf("\n");

    //N to 0 using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i > 0) {
            std::cout << " "; 
        }
    }
    std::cout << std::endl;

    return 0;
}
