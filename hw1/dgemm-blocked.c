#include <printf.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 36
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
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        // For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

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
    // For each column i of transposed A (i.e. row of A)
    for (unsigned int i = 0; i < M; ++i) {
        // For each column j of B
        for (unsigned int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (unsigned int k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

/*
 * Transposes matrix A of size MxM in-place.
 */
void transpose(double* A, int M) {
    // we access a column-major matrix as:
    // A[i,j] = A[row + col * total_columns]
    // For each row i of A
    double temp = 0;
    printf("Before transpose %f, %f, %f, %f, %f, %f, %f, %f, %f", A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]);
    for (unsigned int i = 0; i < M - 1; ++i) {
        // For each column j of A
        for (unsigned int j = i + 1; j < M; ++j) {
            if (i == j) {
                // diagonal elements need not be touched in a transpose operation
                continue;
            }
            // A[1,0] (A[1]) = A[0,1] (A[3])
            printf("Swapping i = %d and j = %d", i, j);
            printf("Swapping A[i,j] = %f and A[j,i] = %f", A[i + j * M], A[j + i * M]);
            // swap A[i,j] and A[j,i]
            temp = A[i + j * M];
            // temp = A[3]
            A[i + j * M] = A[j + i * M];
            // A[3] = A[1]
            A[j + i * M] = temp;
            // A[1] = temp
            printf("Result: A[i,j] = %f, A[j,i] = %f", A[i + j * M], A[j + i * M]);

        }
    }
    printf("After transpose %f, %f, %f, %f, %f, %f, %f, %f, %f", A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7], A[8]);
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices. A is stored in row-major format, B and C are stored in column-major format
 * On exit, A and B maintain their input values. */
void square_dgemm_row_major_A(int lda, double* A, double* B, double* C) {
    // here, we'll transpose A, and after that it'll still be lda x lda
    transpose(A, lda);

    // the big question: does it matter how we iterate the blocks ? I think so.
    // I think it'll be more efficient to iterate the blocks in the direction of the 'majorness' of the matrix.

    // next big question: do we need to adjust m, n, k? i dont think so.

    // then why was M unused in do_block_row_major_A ? because i was dumb

    // For each block-column of A
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

                // the 5th argument in the call below is the top-left entry of the block.
                // is this correct then ?

                // col major matrix :
                // A[i,j] = A[row + col * total_columns]
                do_block_row_major_A(lda, M, N, K, A + k + i * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // here, we'll transpose A, and after that it'll still be lda x lda

    square_dgemm_row_major_A(lda, A, B, C);

    // For each block-row of A
//    for (int i = 0; i < lda; i += BLOCK_SIZE) {
//        // For each block-column of B
//        for (int j = 0; j < lda; j += BLOCK_SIZE) {
//            // Accumulate block dgemms into block of C
//            for (int k = 0; k < lda; k += BLOCK_SIZE) {
//                // Correct block dimensions if block "goes off edge of" the matrix
//                int M = min(BLOCK_SIZE, lda - i);
//                int N = min(BLOCK_SIZE, lda - j);
//                int K = min(BLOCK_SIZE, lda - k);
//                // Perform individual block dgemm
//                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
//            }
//        }
//    }
}
