# Authors: Edouard Oyallon, Muawiz Chaudhary
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna

# def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order,
def scattering2dmul(x, pad, unpad, backend, J, L, phi, psi, max_order,
        out_type='array'):
    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    # Define lists for output.
    out_S_0, out_S_1, out_S_2, out_S_3 = [], [], [], [] # out_S_0, []

    U_r = pad(x)  # J=3: [300, 48, 48, 2] U_r:[300, 40, 40, 2] # U_r:[300, 224+pad, 224+pad, 2]

    U_0_c = fft(U_r, 'C2C') # [48,48,2]

    U_0_c_s0 = subsample_fourier(U_0_c, k=2 ** 0) # [48,48,2]
    U_1_c_s0 = cdgmm(U_0_c_s0, phi[0]) #[48,48,2]

    U_1_c_s1 = subsample_fourier(U_0_c, k=2 ** 1) # [24, 24, 2]
    U_1_c_s1 = cdgmm(U_1_c_s1, phi[1]) # [24,24,2]

    U_1_c_s2 = subsample_fourier(U_0_c, k=2 ** 2) #[12,12,2]
    U_1_c_s2 = cdgmm(U_1_c_s2, phi[2]) #[12, 12, 2]

    # U_1_c_s3 = subsample_fourier(U_1_c, k=2 ** 3) # U_1_c:[300, 5, 5, 2]

    # mid_U_1_c_s0.append({'coef': U_1_c_s0,
    #                      'j': (),
    #                      'theta': ()})
    # mid_U_1_c_s1.append({'coef': U_1_c_s1,
    #                      'j': (),
    #                      'theta': ()})
    # mid_U_1_c_s2.append({'coef': U_1_c_s2,
    #                      'j': (),
    #                      'theta': ()})
    # mid_U_1_c_s3.append({'coef': U_1_c_s3,
    #                      'j': (),
    #                      'theta': ()})
    ## Here could add 3-atten ####

    S_0_s0 = fft(U_1_c_s0, 'C2R', inverse=True) #[300,48,48]
    S_0_s1 = fft(U_1_c_s1, 'C2R', inverse=True) #[300,24,24]
    S_0_s2 = fft(U_1_c_s2, 'C2R', inverse=True) #[300,12,12]
    # S_0 = unpad(S_0) # S_0: [300, 8, 8]

    out_S_0.append({'coef': S_0_s0,
                    'j': (),
                    'theta': ()})  # [300,48,48]*1
    out_S_1.append({'coef': S_0_s1,
                    'j': (),
                    'theta': ()}) # [300,24,24]*1
    out_S_2.append({'coef': S_0_s2,
                    'j': (),
                    'theta': ()}) # [300,12,12]*1

    for n1 in range(len(psi)): # n1=[00]-[07](j=0) [08]-[15](j=1)  [16]-[23](j=2)
        j1 = psi[n1]['j'] # ji = psi[00-07]['j']->0 [48,48,1]/ psi[08-15]['j']->1 [48,48,1][24,24,1] / psi[16-23]['j']->2  [48,48,1] [24,24,1] 0,1,2
        theta1 = psi[n1]['theta'] # theta = 0, 1/8pi,2/8pi,...,7/8pi;

        U_1_c_s0_ = cdgmm(U_0_c, psi[n1][0]) # U_1_c_s0: [300, 48, 48, 2] for 24
        S_1_r_s0 = fft(U_1_c_s0_, 'C2R', inverse=True) # U_1_c: [300, 48, 48] for 24 High_1_s0
        out_S_0.append({'coef': S_1_r_s0,
                        'j': (j1,),
                        'theta': (theta1,)}) #[300,48,48] *(1+ for 24)  High_1_s0

        if j1 > 0:
            U_1_c_s1_ = subsample_fourier(U_0_c, k=2 ** 1) # [24, 24, 2] for 16
            U_1_c_s1_ = cdgmm(U_1_c_s1_, psi[n1][1])
            U_1_c_s2_ = subsample_fourier(U_1_c_s1_, k=2 ** 1) # [12, 12, 2] for 16
            S_1_r_s1 = fft(U_1_c_s1_, 'C2R', inverse=True) # U_1_c: [300, 24, 24] for 16 High_1_s1
            S_1_r_s2 = fft(U_1_c_s2_, 'C2R', inverse=True) # U_1_c: [300, 12, 12] for 16 High_1_s2

            out_S_1.append({'coef': S_1_r_s1,
                            'j': (j1,),
                            'theta': (theta1,)})  # [300,24,24] *(1+ for 16) High_1_s1
            out_S_2.append({'coef': S_1_r_s2,
                            'j': (j1,),
                            'theta': (theta1,)})  # [300,12,12] *(1+ for 16) High_1_s2

        # U_1_c_s3 = subsample_fourier(U_1_c, k=2 ** 4) #14 for 16 High

        # mid_U_1_c_s0.append({'coef': U_1_c_s0,
        #                 'j': (j1,),
        #                 'theta': (theta1,)})
        # mid_U_1_c_s1.append({'coef': U_1_c_s1,
        #                 'j': (j1,),
        #                 'theta': (theta1,)})
        # mid_U_1_c_s2.append({'coef': U_1_c_s2,
        #                 'j': (j1,),
        #                 'theta': (theta1,)})
        # mid_U_1_c_s3.append({'coef': U_1_c_s3,
        #                 'j': (j1,),
        #                 'theta': (theta1,)})
        ## Here to add 3-Atten #######

        # First High pass filter

        # Second low pass filter
        U_1_c_s0_ = modulus(U_1_c_s0_)
        U_1_c_s0_ = fft(U_1_c_s0_, 'C2C') # 48,48,2
        S_2_c_s0 = cdgmm(U_1_c_s0_, phi[0])  # 48,48,2
        S_2_c_s0 = subsample_fourier(S_2_c_s0, k=2 ** 1) #  24, 24,2
        S_2_r_s0 = fft(S_2_c_s0, 'C2R', inverse=True) # 24,24
        out_S_1.append({'coef': S_2_r_s0,
                        'j': (j1,),
                        'theta': (theta1,)}) # 24,24 (1 + for 16 + for 24)

        if j1 > 0:
            U_1_c_s1_ = modulus(U_1_c_s1_)
            U_1_c_s1_ = fft(U_1_c_s1_, 'C2C')  # 24,24,2
            S_2_c_s1 = cdgmm(U_1_c_s1_, phi[1])  # 24,24,2
            S_2_c_s1 = subsample_fourier(S_2_c_s1, k=2 ** 1) # 12,12,2
            S_2_r_s1 = fft(S_2_c_s1, 'C2R', inverse=True) #  12, 12
            out_S_2.append({'coef': S_2_r_s1,
                            'j': (j1,),
                            'theta': (theta1,)})  # [300,28,28] (1+ for 16+ for 16)

        # # S_1_r = unpad(S_1_r) #  S_1_c: [300, 8, 8]

        if max_order < 2:
            continue
        for n2 in range(len(psi)):
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1: # when j1=0, j2=0 continue, for j2=1,2; when j1=1, for j2=2;
                continue

            U_2_c_s0_ = cdgmm(U_1_c_s0_, psi[n2][0])  # 48,48,2
            U_2_c_s0_ = subsample_fourier(U_2_c_s0_, k=2)  # U_2_c: [300, 24, 24, 2]
            U_2_c_s0_ = fft(U_2_c_s0_, 'C2C', inverse=True)
            U_2_c_s0_ = modulus(U_2_c_s0_) # U_2_c: [300, 24, 24, 2]
            U_2_c_s0_ = fft(U_2_c_s0_, 'C2C') # U_2_c: [300, 24 , 24, 2]
            # Third low pass filter
            S_3_c_s0 = cdgmm(U_2_c_s0_, phi[1])  #
            # S_3_c_s0 = subsample_fourier(S_3_c_s0, k=2)  # U_2_c: [300, 24, 24, 2]
            S_3_r_s0 = fft(S_3_c_s0, 'C2R', inverse=True)  # 24,24 for 8*8*3
            out_S_1.append({'coef': S_3_r_s0,
                            'j': (j1, j2),
                            'theta': (theta1, theta2)}) # 24,24 (1 + for 16 + for 24 + for 8*8*3=192)
            if j1 > 0:
                U_2_c_s1_ = cdgmm(U_1_c_s1_, psi[n2][1])  # 24,24,2
                U_2_c_s1_ = subsample_fourier(U_2_c_s1_, k=2)
                U_2_c_s1_ = fft(U_2_c_s1_, 'C2C', inverse=True)
                U_2_c_s1_ = modulus(U_2_c_s1_) # 12,12,2
                U_2_c_s1_ = fft(U_2_c_s1_, 'C2C')  #12,12,2
                # Third low pass filter
                S_3_c_s1 = cdgmm(U_2_c_s1_, phi[2])  # 12,12,2
                # S_2_c_s1 = subsample_fourier(S_2_c_s1, k=2)  # S_2_c: [300, 28, 28, 2]
                S_3_r_s1 = fft(S_3_c_s1, 'C2R', inverse=True)  # 12,12 for 8*8
                out_S_2.append({'coef': S_3_r_s1,
                                'j': (j1, j2),
                                'theta': (theta1, theta2)})

            #S_2_r = unpad(S_2_r) # S_2_r: [300, 8, 8]
    # out_S = []
    # out_S.extend(out_S_0)
    # out_S.extend(out_S_1)
    # out_S.extend(out_S_2)

    if out_type == 'array':
        out_S_0 = concatenate([x['coef'] for x in out_S_0])
        out_S_1 = concatenate([x['coef'] for x in out_S_1])
        out_S_2 = concatenate([x['coef'] for x in out_S_2])
        # out_S_3 = concatenate([x['coef'] for x in out_S_3])
    # return out_S
    # return out_S_1, out_S_2, out_S_3
    return out_S_0, out_S_1, out_S_2


#__all__ = ['scattering2d']
__all__ = ['scattering2dmul']
