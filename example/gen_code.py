import os

float4char = {0: 'x', 1: 'y', 2: 'z', 3: 'w'}

insert = {
  (0, 31): ['loadX0 = tex1Dfetch(tex, track0); // load next strip to register'],
  (0, 33): ['loadX2 = tex1Dfetch(tex, track2); // load next strip to register'],
  (1, 31): ['loadX4 = tex1Dfetch(tex, track4); // load next strip to register'],
  (1, 33): ['loadX6 = tex1Dfetch(tex, track6); // load next strip to register'],
  (5, 30): ['share[writeS + 0*16] = loadX0; // store register to shared memory'],
  (5, 34): ['share[writeS + 2*16] = loadX2; // store register to shared memory'],
  (6, 30): ['share[writeS + 4*16] = loadX4; // store register to shared memory'],
  (6, 34): ['share[writeS + 6*16] = loadX6; // store register to shared memory'],
  (6, 62): ['__syncthreads(); // sync till next strip is stored in shared memory',
            'readAs ^= 128; // togger readAs to read next A strip',
            'readBs ^= 128; // togger readBs to read next B strip',
            'writeS ^= 128; // togger writeS to write to the other shared memory buffer',],
  (7, 63): ['track0 += ldx8;',
            'track2 += ldx8;',
            'track4 += ldx8;',
            'track6 += ldx8;'],
}


indent = '    '
for k in range(8):
  odd = k & 1
  nOdd = not odd
  rsOffset = (k + 1) % 8
  insert[(k, 0)] = ['rA[%d][0] = shareA[readAs + %d*16 + 0]; // load smem to regs' % \
                    (nOdd, rsOffset)]
  insert[(k, 2)] = ['rB[%d][0] = shareB[readBs + %d*16 + 0]; // load smem to regs' % \
                    (nOdd, rsOffset)]
  insert[(k, 4)] = ['rA[%d][1] = shareA[readAs + %d*16 + 8]; // load smem to regs' % \
                    (nOdd, rsOffset)]
  insert[(k, 6)] = ['rB[%d][1] = shareB[readBs + %d*16 + 8]; // load smem to regs' % \
                    (nOdd, rsOffset)]

  print('%s// Iter k = %d' % (indent, k))
  for c in range(64):
    x = c % 8
    xi = x / 4
    xc = float4char[x % 4]
    y = c / 8
    yi = y / 4
    yc = float4char[y % 4]
    print('%srC[%2d] = fma(rA[%d][%d].%s, rB[%d][%d].%s, rC[%d]);' % 
          (indent, c, odd, xi, xc, odd, yi, yc, c))
    if (k, c) in insert:
      for line in insert[(k, c)]:
        print('%s%s' % (indent, line))


indent = '  '
for c in range(64):
  x = c % 8
  y = c / 8
  if x > 3: x += 28
  if y > 3: y += 28
  print('%sC[%2d + %2d*LDC] = rC[%2d] * alpha;' % (indent, x, y, c))

