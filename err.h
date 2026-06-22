#if !defined(__ERROR_H__)
    #define __ERROR_H__
    #include <stdlib.h>

        #define DEBUG
        
        #if defined(DEBUG)
            #define CHECK_ERROR(err) {if (err != cudaSuccess){                                      \
                    printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
                    exit(EXIT_FAILURE);                                                             \
                }}
        #else
            #define CHECK_ERROR(err) {err;}
        #endif
#endif