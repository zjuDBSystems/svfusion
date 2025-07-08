import struct
import os

def fvecs_to_fbin(fvecs_path, fbin_path):
    """
    Converts an fvecs file to an fbin file.

    Args:
        fvecs_path: Path to the input fvecs file.
        fbin_path: Path to the output fbin file.
    """

    num_vectors = 1000000  # 1M
    dimension = 128

    with open(fvecs_path, 'rb') as fvecs_file, open(fbin_path, 'wb') as fbin_file:
        # Write the header to the fbin file (num_vectors, dimension)
        fbin_file.write(struct.pack('<ii', num_vectors, dimension))

        for i in range(num_vectors):
            # Read the dimension (int) from the fvecs file
            d_bytes = fvecs_file.read(4)
            if not d_bytes:
                raise EOFError(f"Unexpected end of file at vector {i}")
            d = struct.unpack('<i', d_bytes)[0]

            # Verify the dimension
            if d != dimension:
                raise ValueError(f"Dimension mismatch at vector {i}: expected {dimension}, got {d}")

            # Read the vector data (float32) from the fvecs file
            vector_data = fvecs_file.read(4 * dimension)
            if not vector_data:
                raise EOFError(f"Unexpected end of file at vector {i}")

            # Write the vector data to the fbin file
            fbin_file.write(vector_data)

            # Optional: Print progress
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} vectors...")

    print(f"Successfully converted {fvecs_path} to {fbin_path}")


# Example usage:
fvecs_file_path = '/data1/pyc/data/sift1M/sift_base.fvecs'  # Replace with your fvecs file path
fbin_file_path = '/data1/pyc/data/sift1M/sift_base.fbin'    # Replace with desired fbin file path

# 确保输出目录存在
os.makedirs(os.path.dirname(fbin_file_path), exist_ok=True)

fvecs_to_fbin(fvecs_file_path, fbin_file_path)