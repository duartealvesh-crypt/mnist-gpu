#include "mnist.h"

// make a 32-bit integer from the 4 input bytes
uint32_t make_uint32(byte buffer[])
{
    return ((uint32_t) buffer[0] << 24) | ((uint32_t) buffer[1] << 16) | ((uint32_t) buffer[2] << 8) | (uint32_t) buffer[3];
}

byte* read_labels(const char filename[], unsigned* n )
{
    FILE* data = fopen(filename, "r");

    if (data == NULL)
    {
        perror(filename);
        exit(1);
    }

    byte buf[4];

    fread(buf, 1, 4, data); // magic number (discarded)
    fread(buf, 1, 4, data); // number of labels
    *n = make_uint32(buf);

    byte* ls = (byte*) calloc(*n, sizeof(byte));

    // Read n labels
    fread(ls, 1, *n, data);

    fclose(data);

    return ls;
}

image* read_images(const char filename[], unsigned* n )
{
    FILE* data = fopen(filename, "r");

    if (data == NULL)
    {
        perror(filename);
        exit(1);
    }

    byte buf[4];

    fread(buf, 1, 4, data); // magic number (discarded)
    fread(buf, 1, 4, data); // number of images
    *n = make_uint32(buf);

    fread(buf, 1, 4, data); // rows (discarded)
    fread(buf, 1, 4, data); // columns (discarded)

    image* is = (image*) calloc(*n, sizeof(image));

    // Read n images
    fread(is, 28*28, *n, data);

    fclose(data);

    return is;
}
