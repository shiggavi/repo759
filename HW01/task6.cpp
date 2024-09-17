#include <iostream>
#include <cstdlib> // For std::atoi
#include <cstdio>  // For printf

int main(int argc, char* argv[]) {
    // Check if the user has provided a command line argument
    if (argc != 2) {
        std::cerr << "Usage: ./task6 N\n";
        return 1;
    }

    // Convert the command line argument to an integer
    int N = std::atoi(argv[1]);

    // Print from 0 to N using printf
    for (int i = 0; i <= N; ++i) {
        printf("%d", i);
        if (i < N) {
            printf(" "); // Print space between numbers
        }
    }
    printf("\n");

    // Print from N to 0 using std::cout
    for (int i = N; i >= 0; --i) {
        std::cout << i;
        if (i > 0) {
            std::cout << " "; // Print space between numbers
        }
    }
    std::cout << std::endl;

    return 0;
}
