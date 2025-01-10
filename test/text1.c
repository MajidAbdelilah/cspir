/* Simple vectorizable loop */
void simple_loop(float* arr, int n) {
    int i;
    for(i = 0; i < n; i++) {
        arr[i] = arr[i] * 2.0f;
    }
}

/* Loop with reduction */
float reduction_loop(float* arr, int n) {
    int i;
    float sum = 0.0f;
    for(i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

/* Loop with dependency (non-vectorizable) */
void dependency_loop(float* arr, int n) {
    int i;
    for(i = 1; i < n; i++) {
        arr[i] = arr[i-1] + 1.0f;
    }
}

/* Loop with mixed types (should flag warning) */
void mixed_types_loop(float* arr, int* iarr, int n) {
    int i;
    for(i = 0; i < n; i++) {
        arr[i] = (float)iarr[i];
    }
}

/* Loop with constant trip count */
void constant_loop(float* arr) {
    int i;
    for(i = 0; i < 128; i++) {
        arr[i] = arr[i] + 1.0f;
    }
}

int main(void) {
    float arr[128];
    int iarr[128];
    float sum;

    simple_loop(arr, 128);
    sum = reduction_loop(arr, 128);
    dependency_loop(arr, 128);
    mixed_types_loop(arr, iarr, 128);
    constant_loop(arr);

    return (int)sum;
}
