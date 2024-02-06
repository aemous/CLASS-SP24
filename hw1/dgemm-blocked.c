#include <immintrin.h>
#include <stdio.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 16
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // TODO one might consider static allocating the blocks and dotProduct vector in advanced
    double* AT = _mm_malloc(K * M * sizeof(double), 64); // K rows, M columns
    double* dotProduct = _mm_malloc(4 * sizeof(double), 64); // temp array of size 4 whose sum is a dot product
    __m256d rowA1; // stores first quarter of row i of A
    __m256d rowA2; // stores second quarter of row i of A
    __m256d rowA3; // stores third quarter of row i of A
    __m256d rowA4; // stores fourth quarter of row i of A
    __m256d colB1; // stores first quarter of column j of B
    __m256d colB2; // stores second quarter of column j of B
    __m256d colB3; // stores third quarter of column j of B
    __m256d colB4; // stores fourth quarter of column j of B

    // transpose the A-block for SIMD-compatibility
    // For each column j of AT
    for (unsigned int j = 0; j < M; ++j) {
        // For each row i of AT
        for (unsigned int i = 0; i < K; ++i) {
            AT[i + j * K] = A[j + i * lda];
        }
    }

    // compute the multiplication
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            // TODO if we get correctness error, it might be due to the case that the block is smaller than BLOCK_SIZE
            rowA1 = _mm256_load_pd(AT + i * K);
            rowA2 = _mm256_load_pd(AT + 4 + i * K);
            rowA3 = _mm256_load_pd(AT + 8 + i * K);
            rowA4 = _mm256_load_pd(AT + 12 + i * K);
            colB1 = _mm256_load_pd(B + j * lda);
            colB2 = _mm256_load_pd(B + 4 + j * lda);
            colB3 = _mm256_load_pd(B + 8 + j * lda);
            colB4 = _mm256_load_pd(B + 12 + j * lda);

            // compute first 'half' of the dot product of A[i,:] and B[:,j]
            __m256d dot1 = _mm256_hadd_pd(_mm256_mul_pd(rowA1, colB1), _mm256_mul_pd(rowA2, colB2));
            // compute second 'half' of the dot product of A[i,:] and B[:,j]
            __m256d dot2 = _mm256_hadd_pd(_mm256_mul_pd(rowA3, colB3), _mm256_mul_pd(rowA4, colB4));
//
//            // the sum of the 4 doubles in the vector below is the dot product of A[i,:] and B[:,j]
            _mm256_store_pd(dotProduct, _mm256_hadd_pd(dot1, dot2));
            double cij = C[i + j * lda];
            for (int k = 0; k < 4; k++) {
                cij += dotProduct[k];
            }
            C[i + j * lda] += cij;
        }
    }
    free(AT);
    free(dotProduct);
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
                printf("i = %d, j = %d, k = %d", i, j ,k);
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
