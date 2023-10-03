#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <complex>
#include <vector>
#include <math.h>

// complex input: analyzing all 3 dims for both R2C and C2C transforms
void Get_Frequency_Spectrum(int IsR2C, size_t inx, size_t iny, size_t inz, std::complex<float> sx[])
{
    size_t offset,doffset;
    size_t i,j,k;
    size_t max_indx;

    // deal with contracted leading (ie. inner) index
    size_t c_inx,l_inx;
    if (IsR2C == 1)
      {
        c_inx = inx/2 + 1; // length of x-dir in wavenumber space
        l_inx = c_inx;     // max index for analyzing in x-dir
      }
    else
      {
        c_inx = inx;       // length of x-dir in wavenumber space
        l_inx = inx/2;     // max index for analyzing in x-dir
      }

    max_indx = c_inx;
    if (iny > max_indx) max_indx = iny;
    if (inz > max_indx) max_indx = inz;

    float anx = float(inx);
    float any = float(iny);
    float anz = float(inz);
    float factor = 1.0/(anx*any*anz);

    float Fs;
    float mag_sx;

    float *spec_freq = NULL;
    float *spec_amp = NULL;

    spec_freq = (float*)malloc(3*max_indx*sizeof(float));
    spec_amp = (float*)malloc(3*max_indx*sizeof(float));

    std::cout << std::scientific << std::setprecision(5);

    // re-scale sx by factor 
    // NOTE: this replaces the scaling that would normally 
    //       be performed in the frequency analysis
    for (k = 0; k < inz; k++)
      {
        for (j = 0; j < iny; j++)
          {
            for (i = 0; i < c_inx; i++)
              {
                offset = k*iny*c_inx + j*c_inx + i;

                sx[offset] *= factor;
              }
          }
      }

    for (i = 0; i < 3*max_indx; i++)
      {
        spec_freq[i] = 0.0;
        spec_amp[i] = 0.0;
      }

    float x_counter, y_counter, z_counter;

    // x-direction with contiguous data
    x_counter = 0.0;
    Fs = anx;
    doffset = 0*max_indx;

    for (k = 0; k < inz; k++)
      {
        for (j = 0; j < iny; j++)
          {
            for (i = 0; i < l_inx; i++)
              {
                offset = k*iny*c_inx + j*c_inx + i;

                // std::cout << real(sx[offset]) << "  " << imag(sx[offset]) << std::endl;

                // compute magnitude of wave-space
                mag_sx = abs(sx[offset]);

                // normally, we would re-scale the value here, 
                // but we're doing this above to remove the accumulated
                // scaling for all dims
                // mag_sx /= anx;

                // account for the power in the "negative" side 
                if (i > 0 && i < (c_inx - 1))
                  {
                    mag_sx *= 2.0;
                  }

                // accumulate the value
                spec_amp[i] += mag_sx;

                // since sampling rate = length, Fs/anx = 1.0
                // freq = float(i)*Fs/anx;
                spec_freq[i] += float(i);
              }
            x_counter += 1.0;
          }
      }

    // Now for the y-direction
    // switch the i and j loops to collect the spectrum details in the 
    // y-direction (ie. strided)
    y_counter = 0.0;
    Fs = any;
    doffset = 1*max_indx;

    for (k = 0; k < inz; k++)
      {
        for (i = 0; i < c_inx; i++)
          {
            for (j = 0; j < iny/2; j++)
              {
                // use the correct offset calculation regardless of the 
                // strided form of these loops
                offset = k*iny*c_inx + j*c_inx + i;

                // std::cout << real(sx[offset]) << "  " << imag(sx[offset]) << std::endl;

                // compute magnitude of wave-space
                mag_sx = abs(sx[offset]);

                // normally, we would re-scale the value here, but 
                // we're doing this above to remove the accumulated
                // scaling for all dims
                // mag_sx /= axl;

                // account for the power in the "negative" side 
                if (j > 0 && j < (iny - 1))
                  {
                    mag_sx *= 2.0;
                  }

                spec_amp[doffset + j] += mag_sx;

                // since sampling rate = length, Fs/any = 1.0
                // freq = float(i)*Fs/any;
                spec_freq[doffset + j] += float(j);
              }
            y_counter += 1.0;
          }
      }

    // Now for the z-direction
    // switch the i and k loops to collect the spectrum details in the 
    // z-direction (ie. strided)
    z_counter = 0.0;
    Fs = anz;
    doffset = 2*max_indx;

    for (j = 0; j < iny; j++)
      {
        for (i = 0; i < c_inx; i++)
          {
            for (k = 0; k < inz/2; k++)
              {
                // use the correct offset calculation regardless of the 
                // strided form of these loops
                offset = k*iny*c_inx + j*c_inx + i;

                // std::cout << real(sx[offset]) << "  " << imag(sx[offset]) << std::endl;

                // compute magnitude of wave-space
                mag_sx = abs(sx[offset]);

                // normally, we would re-scale the value here, 
                // but we're doing this above to remove the accumulated
                // scaling for all dims
                // mag_sx /= axl;

                // account for the power in the "negative" side 
                if (k > 0 && k < (inz - 1))
                  {
                    mag_sx *= 2.0;
                  }

                spec_amp[doffset + k] += mag_sx;

                // since sampling rate = length, Fs/axl = 1.0
                // freq = float(i)*Fs/axl;
                spec_freq[doffset + k] += float(k);
              }
            z_counter += 1.0;
          }
      }


    // post-process and clean up the data for the R2C transforms
    // NOTE: needed since the leading index is contracted 
    if (IsR2C == 1)
      {
        float val[3];

        // first, collect the non-constant amplitudes
        for (j = 0; j < 3; j++)
          {
            val[j] = 0.0;
            for (i = 1; i < max_indx; i++)
              {
                val[j] += spec_amp[j*max_indx + i];
              }
            // std::cout << val[j] << std::endl;
          }

        // find the dc value
        float dcval, dcx, dcy, dcz;
        dcx = spec_amp[0] - val[1] - val[2];
        dcy = spec_amp[max_indx + 0] - 0.5*val[0] - val[2];
        dcz = spec_amp[2*max_indx + 0] - 0.5*val[0] - val[1];
        dcval = (dcx + dcy + dcz)/3.0;
        // std::cout << dcval << " " << dcx << " " << dcy << " " << dcz << std::endl;
        // std::cout << fabs(dcval - dcx) << " " << fabs(dcval - dcy) << " " << fabs(dcval - dcz) << std::endl;

        // correct the dc values in the y and z directions 
        spec_amp[max_indx + 0] = dcval + val[0] + val[2];
        spec_amp[2*max_indx + 0] = dcval + val[0] + val[1];
      }

    // time to write the data
    std::ofstream outfile;

    outfile.open("x_spectrum", std::ios::out);
    outfile << std::scientific << std::setprecision(5);
    doffset = 0*max_indx;
    for (i = 0; i < l_inx; i++)
      {
        // silly sanity check with a simple average to make sure all x-values are set
        spec_freq[i] /= x_counter;

        // spec_amp[i] /= factor;
        outfile << spec_freq[i] << "  " << spec_amp[i] << "  "  << std::endl;
      }
    outfile.close();

    outfile.open("y_spectrum", std::ios::out);
    outfile << std::scientific << std::setprecision(5);
    doffset = 1*max_indx;
    for (j = 0; j < iny/2; j++)
      {
        // silly sanity check with a simple average to make sure all y-values are set
        spec_freq[doffset + j] /= y_counter;

        // spec_amp[j] /= factor;
        outfile << spec_freq[doffset + j] << "  " << spec_amp[doffset + j]  << std::endl;
      }
    outfile.close();

    outfile.open("z_spectrum", std::ios::out);
    outfile << std::scientific << std::setprecision(5);
    doffset = 2*max_indx;
    for (k = 0; k < inz/2; k++)
      {
        // silly sanity check with a simple average to make sure all z-values are set
        spec_freq[doffset + k] /= z_counter;

        // spec_amp[k] /= factor;
        outfile << spec_freq[doffset + k] << "  " << spec_amp[doffset + k]  << std::endl;
      }
    outfile.close();

    // cleanup
    free(spec_freq);
    free(spec_amp);

    return;
}


