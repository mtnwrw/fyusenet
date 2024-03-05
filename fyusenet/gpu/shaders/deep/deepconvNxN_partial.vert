/* ----------------------------------------------------------------------------
 * NxN partial conv vertex shader (deep-format) Copyright (c) 2023 Fyusion Inc.
 * Creator: Martin Wawro
 * SPDX-License-Identifier: MIT
 * ------------------------------------------------------------------------- */

precision highp float;
precision highp int;
precision highp sampler2D;

#ifdef BINDING_SUPPORT
layout(binding=4) uniform highp sampler2D inputDisplacements;
#ifdef NO_HALF
layout(binding=5) uniform sampler2D inputCoeffs;
#else
layout(binding=5) uniform highp usampler2D inputCoeffs;
#endif
#else
uniform highp sampler2D inputDisplacements;
#ifdef NO_HALF
uniform sampler2D inputCoeffs;
#else
uniform highp usampler2D inputCoeffs;
#endif
#endif

in highp vec4 attributes0;
in highp ivec2 attributes1;
in highp vec2 attributes2;

out highp vec4 texCoord;
#ifdef NO_HALF
flat out mediump vec4 layer0coeffs[COEFF_VARYINGS];
#else
flat out highp uvec4 layer0coeffs[COEFF_VARYINGS];
#endif

#ifdef USE_RESIDUAL
out highp vec2 resCoord;
#endif

uniform int instancesPerTile;
uniform int horizOffset;
uniform int kernelOffset;
uniform int numParts;
uniform float textureStep;

void main(void) {
  gl_Position = vec4(attributes0.x, attributes0.y, 0.0, 1.0);
  texCoord = vec4(attributes0.z, attributes0.w, 0.0, 0.0);
#ifdef INSTANCE_OFFSET
  int instance = gl_InstanceID + INSTANCE_OFFSET;
#else
  int instance = gl_InstanceID;
#endif
  // Displace texture coordinates of input tile for kernel window shift (horiz + vert)
  int intile = instance / instancesPerTile;
  int vkernelidx = instance % instancesPerTile;
  texCoord.xy += texelFetch(inputDisplacements, ivec2(intile, vkernelidx),0).rg;
  texCoord.x += float(horizOffset) * textureStep;
#ifdef NO_HALF
  int inchan = 4 * intile * KERNEL + kernelOffset * 4;
#else
  int inchan = 2 * intile * KERNEL + kernelOffset * 2;
#endif
  int ybase = vkernelidx + attributes1.x;
  // fetch weights
#ifdef NO_HALF
  for (int i=0; i < COEFF_VARYINGS; i++) {
    layer0coeffs[i] = texelFetch(inputCoeffs, ivec2(inchan+i, ybase),0);
  }
#else
  for (int i=0; i < COEFF_VARYINGS; i++) {
    layer0coeffs[i] = texelFetch(inputCoeffs, ivec2(inchan+i, ybase),0);
  }
#endif
#if !defined(NO_BIAS) || defined(POST_BATCHNORM)
  if (instance == 0) {
    texCoord.z = float(attributes1.y+1);
  } else {
    texCoord.z = 0.0;
  }
#else
  texCoord.z = 0.0;
#endif
  texCoord.w = float(attributes1.y+1);
#ifdef USE_RESIDUAL
  resCoord.xy = attributes2.xy;
#endif
}

