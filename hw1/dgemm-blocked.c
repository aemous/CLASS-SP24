#include <stdlib.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 20
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * So, it looks like this implementation basically does the naive dgemm, except on smaller
 * blocks instead of the whole matrix.
 */

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block_row_major_A(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // transpose the KxM A-block by copying to a new memory location (ideally, aligned)
    // swap instances of A below with transposed version

    double* AT = malloc(K * M * sizeof(double)); // K rows, M columns
    // For each column j of AT
    for (unsigned int j = 0; j < M; ++j) {
        // For each row i of AT
        for (unsigned int i = 0; i < K; ++i) {
            AT[i + j * K] = A[j + i * lda];
        }
    }

//    printf("AT: \n");
//    // For each row i of AT
//    for (unsigned int i = 0; i < K; ++i) {
//        // For each column j of AT
//        for (unsigned int j = 0; j < M; ++j) {
//            printf("%f ", AT[i + j * K]);
//        }
//        printf("\n");
//    }

    // For each column i of transposed A (i.e. row of A)
    for (unsigned int i = 0; i < M; ++i) {
        // For each column j of B
        for (unsigned int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (unsigned int k = 0; k < K; ++k) {
                cij += AT[k + i * K] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }

    // free the transposed A from memory
    free(AT);
}

/*
 * Transposes matrix A of size MxM in-place.
 */
void transpose(double* A, int M) {
    // we access a column-major matrix as:
    // A[i,j] = A[row + col * total_rows]
    // For each row i of A
    double temp = 0;
    for (unsigned int i = 0; i < M - 1; ++i) {
        // For each column j of A
        for (unsigned int j = i + 1; j < M; ++j) {
            if (i == j) {
                // diagonal elements need not be touched in a transpose operation
                continue;
            }
            temp = A[i + j * M];
            A[i + j * M] = A[j + i * M];
            A[j + i * M] = temp;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // here, we'll transpose A, and after that it'll still be lda x lda
//    transpose(A, lda);
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);

                // col major matrix :
                // A[i,j] = A[row + col * total_rows]
                // TODO if we fail correctness, swap k and i in A arg below
                do_block_row_major_A(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
//    transpose(A, lda);
}
