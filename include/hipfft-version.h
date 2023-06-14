#pragma once

#if 0

// will eventually use this which was 
// replicated from h4i-hipblas and h4i-hipsolver
#define hipfftVersionMajor @PROJECT_VERSION_MAJOR@
#define hipfftVersionMinor @PROJECT_VERSION_MINOR@
#define hipfftVersionPatch @PROJECT_VERSION_PATCH@
#define hipfftVersionTweak 0 // We don't use a tweak

#else

// basic definition for initial development and testing
#define hipfftVersionMajor 1
#define hipfftVersionMinor 0
#define hipfftVersionPatch 10
#define hipfftVersionTweak 0

#endif

