// bitonic sort within a single thread, see:
// https://github.com/facebookresearch/HVVR/blob/master/libraries/hvvr/raycaster/sort.h
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/Data-Parallel_Algorithms.html ("Bitonic Sort" sample, bitonic_kernel.cu)
// https://github.com/Microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/ParticleTileCullingCS.hlsl
// http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
// If count is known at compile time, this should be optimized down to a series
// of min/max ops without branching.
void sortBitonic(inout SORT_T sortKeys[SORT_SIZE])
{
    int count = SORT_SIZE;

	// find the power of two upper bound on count to determine how many iterations we need
    int countPow2 = count;
    if (countbits(count) > 1) {
        countPow2 = (1 << (32 - firstbithigh(count)));
    }

    for (int k = 2; k <= countPow2; k *= 2) {
        // align up to the current power of two
        count = (count + k - 1) & ~(k - 1);

		// merge
        for (int j = k / 2; j > 0; j /= 2) {
			// for each pair of elements
            for (int i = 0; i < count / 2; i++) {
				// find the pair of elements to compare
                int mask = j - 1;
                int s0 = ((i & ~mask) << 1) | (i & mask);
                int s1 = s0 | j;

                SORT_T a = sortKeys[s0];
                SORT_T b = sortKeys[s1];

                bool compare = SORT_CMP_LESS(a, b);
                bool direction = ((s0 & k) == 0);

                if (compare != direction) {
                    sortKeys[s0] = b;
                    sortKeys[s1] = a;
                }
            }
        }
    }
}
