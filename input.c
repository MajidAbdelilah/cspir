void vector_add(float* input, float* output, int size) {
    for(int i = 0; i < size; i++) {
        output[i] = input[i] * 2.0f;
    }
}
