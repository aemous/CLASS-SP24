#include <immintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // Here's the plan for SIMD:
    // Switch block size to 8
    // Declare 2 512 vectors, to store a row of A and a column of B
    double* AT = _mm_malloc(K * M * sizeof(double), 64); // K rows, M columns
    // For each column j of AT
    for (unsigned int j = 0; j < M; ++j) {
        // For each row i of AT
        for (unsigned int i = 0; i < K; ++i) {
            AT[i + j * K] = A[j + i * lda];
        }
    }

    __m512d rowA;
    __m512d colB;

    // Then, replace the inner-most loop with:
        // Load row i of A into the 512 vector
        // Load column j of B into the 512 vector
        // call _mm512_mul_pd on the two vectors, then call
        // _mm512_reduce_add_pd on the result, add this result to cij.

    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            rowA = _mm512_load_pd(AT + i * K);
            colB = _mm512_load_pd(B + j * K);
//            double cij = C[i + j * lda];
//          cij += A[i + k * lda] * B[k + j * lda];
            C[i + j * lda] += _mm512_reduce_add_pd(_mm512_mul_pd(rowA, colB));
        }
    }
    free(AT);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
           }
        }
    }
}
