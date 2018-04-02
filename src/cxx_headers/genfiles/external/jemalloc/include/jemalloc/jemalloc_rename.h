/*
 * Name mangling for public symbols is controlled by --with-mangling and
 * --with-jemalloc-prefix.  With default settings the je_ prefix is stripped by
 * these macro definitions.
 */
#ifndef JEMALLOC_NO_RENAME
#  define je_free jemalloc_free
#  define je_malloc jemalloc_malloc
#  define je_posix_memalign jemalloc_posix_memalign
#  define je_realloc jemalloc_realloc
#endif
