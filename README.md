# N Dimensional Lomb Scargle Periodogram

## Description
The Lomb-Scargle Periodogram (LSP) is a very useful numerical tool for spectral analysis. However, it is only supported
in one dimension in scipy at the time of this writing. This code provides a multivariate extension to the Lomb-Scargle
periodogram in python. This code expands upon the original distribution for this
[article (doi:10.3389/fspas.2024.1519436)](https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2024.1519436/full)


## Usage

The following demo illustrates the ND-LSP in 1D.

```python
import matplotlib.pyplot as plt
import numpy as np

import ndlsp as nd

np.random.seed(42)
n = 400                   # number of sample points
f0, phi0, amp0 = 5, 0.2 * np.pi, 3  # frequency, phase, amplitude for sinusoid
t = np.random.rand(n)     # Non-uniform point samples between 0 and 1

y = (
    amp0 * np.sin(2 * np.pi * f0 * t + phi0)    # Sinusoid
    + 0.1 * np.random.randn(n)                  # Noise
)
y_prime = y - np.mean(y)

# now run the LSP
X = np.reshape(t, (1, n))
fsamp = np.linspace(start=0.5, stop=10, num=1000)  # Frequencies in units of [1/X]
A, phi = nd.lsp_nd(X, y_prime, [fsamp])           # Amplitude and phase
recon_freq = fsamp[np.argmax(A)]
recon_phase = 2. * np.pi * (1 / fsamp[np.argmax(A)]) * np.mean(t)

# Plot the reconstruction
f, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex='none')
f: plt.Figure
ax1.plot(t, y, '.', label='Data')
ax1.set_xlabel('Time []')
ax1.set_ylabel('Value []')
ax1.legend(loc="upper right")

ax2.plot(fsamp, A)
ax2.set_xlabel('Frequency []')
ax2.set_ylabel('amplitude []')
ax2.plot([f0, f0], [0, amp0], 'r--', label=f'Known freq = {f0:.2f}')
ax2.plot([recon_freq, recon_freq], [0, amp0], 'r--', label=f'Recon. freq = {recon_freq:.4f}')
ax2.legend()

ax3.plot(fsamp, phi / np.pi)
ax3.set_xlabel('Frequency []')
ax3.set_ylabel(r'Phase [$\pi$]')
ax3.set_ylim([-1.1, 1.1])
ax3.plot([f0, f0], [-1, 1], 'r--', label=r'Known phase: %.2f $\pi$' % (phi0 / np.pi))
ax3.plot([recon_freq, recon_freq], [-1, 1], 'r--', label=r'Recon phase = %.4f $\pi$' % (recon_phase / np.pi))
ax3.legend()
ax3.set_yticks([-1., 0, 1.])
ax3.set_yticks([-1., -0.75, -0.5, -0.25, 0.,  0.25,  0.5,  0.75,  1.0], minor=True)
ax3.grid(which='both')
f.tight_layout()
f.show()
```

### 2D

A 2D example (without the phase) is shown below

```python
n = 2000
np.random.seed(42)
x = np.random.rand(n)
y = np.random.rand(n)
X = np.vstack((x, y))

k1 = np.array([3, 4])
k2 = np.array([-3, 1])
z = (
    1 * np.sin(2 * np.pi * (np.sum(k1[:, None] * X, axis=0)))    # Sinusoid #1
    + 2 * np.sin(2 * np.pi * (np.sum(k2[:, None] * X, axis=0)))  # Sinusoid #2
    + 0.5 * np.random.randn(n)                                   # Random noise
)

# now run the LSP
fx = np.linspace(-5, 5, 40)
fy = np.linspace(-5, 5, 41)
fs = [fx, fy]
A, phi = nd.lsp_nd(X, z, fs)

# plot the results - move this to a function eventually
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
im = ax1.scatter(x, y, c=z)
cb = plt.colorbar(im, ax=ax1)
cb.set_label('Value []')
ax1.set_xlabel('X []')
ax1.set_ylabel('Y []')

im = ax2.contourf(fx, fy, A.T)
cb = plt.colorbar(im, ax=ax2)
cb.set_label('Amplitude []')
ax2.plot(*k1, 'rx', label="Wave #1 known freq",)
ax2.plot(*k2, 'gx', label="Wave #2 known freq",)
ax2.set_xlabel('X frequency []')
ax2.set_ylabel('Y frequency []')
ax2.legend(loc="lower right")
f.tight_layout()
f.savefig("2d.png")
f.show()
```

### Reconstruction

The simplest reconstruction of one wave is facilitated by the `reconstruct` method

```python
xg = np.arange(0, 1.01, 0.01)
yg = np.arange(0, 1.01, 0.01)
grids = (xg, yg)
levs2 = np.arange(-4.0, 4.01, 0.25)

yre_1 = nd.reconstruct(A, phi, fs, [8, 24], grids)

f_recon_one_wave, ax_recon_one_wave = plt.subplots()
im = ax_recon_one_wave.contourf(xg, yg, yre_1.T, levs2, cmap='viridis')
cb = plt.colorbar(im, ax=ax_recon_one_wave)
cb.set_label('Value []')
ax_recon_one_wave.set_title("Wave Reconstruction")
ax_recon_one_wave.set_xlabel('X []')
ax_recon_one_wave.set_ylabel('Y []')
f_recon_one_wave.show()
```

Reconstructing multiple wave is facilitated with the `iterative_orthogonal_reconstruction`
method.

```python
A, phi, inner_prod = nd.lsp_nd(X, z, fs, retrieve_orthogonality=True)
yre_2 = nd.iterative_orthogonal_reconstruction(A, phi, inner_prod, fs,
                                               grids=grids,
                                               ortho_thresh=3e-1)
f_recon_waves, ax_recon_waves = plt.subplots()
im = ax_recon_waves.contourf(xg, yg, yre_2.T, levs2, cmap='viridis')
cb = plt.colorbar(im, ax=ax_recon_waves)
cb.set_label('Value []')
ax_recon_waves.set_title("Waves Reconstruction")
ax_recon_waves.set_xlabel('X []')
ax_recon_waves.set_ylabel('Y []')
f_recon_waves.show()
```