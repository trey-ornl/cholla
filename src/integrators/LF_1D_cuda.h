#pragma once

#include "../global/global.h"

void LF_Algorithm_1D_CUDA(Real *energy, Real *density, Real *momentum, int nx, Real dx, Real dy);
void Free_Memory_LF_1D();
