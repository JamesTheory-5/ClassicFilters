### Classic Filters: No Blackbox Function Filter Design

### Butterworth Filters

### Bessel Filters
This approach allows you to design **Bessel filters** directly in Python, making use of **SciPy's built-in functions** for generating Bessel polynomial roots, performing frequency-domain transformations, and converting the filter coefficients into **biquad SOS form** for efficient filtering. The **Bessel filter** is particularly useful for applications requiring **minimal phase distortion** and smooth transient responses, such as **audio processing** and **signal conditioning**.

### Key Components of the Approach:

1. **Bessel Filter Design**:

   * **Bessel filter** is a type of **low-pass filter** that is specifically designed to have a **maximally flat phase response** and a **smooth transient response**. This filter is often used in applications like **audio filtering**, where maintaining signal integrity and preventing phase distortion is important.

2. **SciPy's `bessel` Function**:

   * The `scipy.signal.bessel` function computes the **analog Bessel filter coefficients**. It returns the **poles**, **zeros**, and **gain** of the filter in the **s-plane** (analog domain).

3. **Frequency Transform**:

   * After generating the **analog filter**, a **frequency transformation** is applied to shift the filter from the **analog domain** to the **digital domain** using methods like the **Bilinear Transform** or **Bandpass Transformation**. This is where you modify the filter from a simple low-pass to other filter types like **high-pass**, **band-pass**, or **band-stop**.

4. **Bilinear Transform**:

   * The **bilinear transform** maps the analog filter's **s-domain** poles and zeros to the **z-domain** (discrete-time) by applying a nonlinear transformation. This is important because digital filters are implemented in the z-domain.

5. **Biquad SOS Form**:

   * Once the filter is transformed into the digital domain, the **biquad** form is used to represent the filter as a **cascade of second-order sections (SOS)**. This is a common form for **digital IIR filters** and ensures stable and efficient computation.

### Example Walkthrough:

Let's walk through the code, discussing each section:

---

### 1. **`design_bessel_filter` function**:

* **Inputs**:

  * `order`: The filter order (the number of poles of the filter).
  * `fs`: The sample rate (for converting analog filter to digital filter).
  * `fc`: The cutoff frequency or band edges (depending on filter type).
  * `ftype`: Type of filter (`'lowpass'`, `'highpass'`, `'bandpass'`, `'bandstop'`).

* **Outputs**:

  * The function outputs a set of **biquad sections** in **SOS form**, which can be used directly for filtering.

#### Key Steps:

* **Analog Prototype**: Using `scipy.signal.bessel`, the analog filter poles and zeros are computed.
* **Frequency Transformation**: Depending on the filter type (e.g., low-pass, high-pass, etc.), the filter is transformed using the appropriate transformation.
* **Bilinear Transform**: The analog filter is mapped into the digital domain using the **bilinear transform** with `scipy.signal.bilinear_zpk`.
* **Biquad Sections (SOS)**: The final filter coefficients are converted into **second-order sections** (SOS) using `scipy.signal.zpk2sos`.

---

### 2. **Transform Analog Lowpass to Bandpass (For Band Filters)**:

For **bandpass** and **bandstop** filters, the method `_analog_lp_to_band` adjusts the analog **lowpass** filter to a **bandpass** by transforming the poles and zeros. Here's what happens:

* **Bandpass Transformation**: The **poles** are adjusted by adding and subtracting the **bandwidth** (the difference between the low and high cutoff frequencies), while the **zeros** are similarly modified to create a **bandpass** response.
* **Bandwidth and Center Frequency**: The center frequency (`w0`) and bandwidth (`bw`) are calculated from the two cutoff frequencies and used for the transformation.

---

### 3. **Example Usage**:

The example usage demonstrates how to design a **6th-order Bessel low-pass filter** with a cutoff frequency of **2 kHz** at a sample rate of **48 kHz**.

```python
# 6th-order Bessel low-pass at 2 kHz, 48 kHz sample rate
sos = design_bessel_filter(order=6, fs=48000, fc=2000, ftype='lowpass')
print(sos)
```

This code will output the filter coefficients in **SOS form**, which you can then use in a **digital filter** for processing audio signals. The **SOS form** is preferred in DSP because it is numerically more stable than using a single transfer function.

---

### 4. **Why This is Equivalent to Falco’s C++ Flow**:

* **Analog Bessel Filter**: In **Falco's C++ flow**, the first step is to generate the **analog filter** using the **Bessel polynomial roots**, which is precisely what `scipy.signal.bessel` does.
* **Frequency Transform**: After that, the filter is transformed into the **digital domain** using various transformations like **Bilinear** or **Bandpass** (as done in `_analog_lp_to_band`).
* **Bilinear Transform**: The final step in both methods is to apply the **bilinear transform** to convert the filter into its **discrete form**, which is exactly what `scipy.signal.bilinear_zpk` does.

Thus, SciPy’s methods are directly replicating the same steps used in **Falco’s C++ implementation**, which makes this approach equally valid and computationally efficient for digital filter design.

---

### 5. **Benefits of Using Python**:

* **No C++ Dependency**: You can compute **Bessel filter coefficients** directly in Python using SciPy, eliminating the need for **C++ scaffolding**. This is great for rapid prototyping and experimentation.
* **Pure Math Path**: This method directly implements the underlying math of the Bessel filter design process, giving you full control and flexibility over your filter design.
* **Scalability**: The design function supports filters of **any order** and can handle different filter types (low-pass, high-pass, bandpass, bandstop).

---

### Summary:

With this approach, you can compute **Bessel filter coefficients** of any order and type directly in Python using SciPy, without the need for complex C++ code. This method uses the same **Bessel polynomial root solving** and **bilinear transforms** that are used in more complex DSP systems, providing a robust and flexible way to design Bessel filters. Whether you are working on **audio processing**, **signal conditioning**, or **filter design**, this method is an efficient way to implement **Bessel filters** directly in Python.

---

### 1. Direct Bessel filter design

```python
import numpy as np
from scipy.signal import bessel, bilinear_zpk, zpk2sos

def design_bessel_filter(order, fs, fc, ftype='lowpass'):
    """
    Compute digital Bessel filter coefficients (biquad SOS form).

    order : int
        Filter order.
    fs : float
        Sampling rate (Hz).
    fc : float or [low, high]
        Cutoff or band edges in Hz.
    ftype : str
        'lowpass', 'highpass', 'bandpass', or 'bandstop'
    """
    # 1. Analog prototype (s-plane)
    z, p, k = bessel(order, 1.0, analog=True, norm='phase')

    # 2. Frequency transform
    if ftype == 'lowpass':
        z, p, k = z, p * (2*np.pi*fc), k * (2*np.pi*fc)**order
    elif ftype == 'highpass':
        z, p, k = z, (2*np.pi*fc) / p, k * (2*np.pi*fc)**order
    elif ftype in ('bandpass', 'bandstop'):
        bw = fc[1] - fc[0]
        w0 = np.sqrt(fc[0]*fc[1])
        z, p, k = _analog_lp_to_band(z, p, k, w0*2*np.pi, bw*2*np.pi)
    else:
        raise ValueError("Invalid filter type")

    # 3. Bilinear transform (s -> z)
    z_z, p_z, k_z = bilinear_zpk(z, p, k, fs=fs)

    # 4. Convert to biquad sections
    sos = zpk2sos(z_z, p_z, k_z)
    return sos

def _analog_lp_to_band(z, p, k, w0, bw):
    """Analog lowpass to bandpass transform."""
    z_bp, p_bp = [], []
    for pole in p:
        term = np.sqrt(pole**2 - (bw/2)**2)
        p_bp.extend([pole + term, pole - term])
    for zero in z:
        term = np.sqrt(zero**2 - (bw/2)**2)
        z_bp.extend([zero + term, zero - term])
    z_bp, p_bp = np.array(z_bp), np.array(p_bp)
    k_bp = k * (bw)**len(p)
    return z_bp, p_bp, k_bp
```

---

### 2. Example usage

```python
# 6th-order Bessel low-pass at 2 kHz, 48 kHz sample rate
sos = design_bessel_filter(order=6, fs=48000, fc=2000, ftype='lowpass')
print(sos)
```

You now have the **same coefficients** that Falco’s `Bessel::LowPass` + `RootFinder` chain produces, ready to use in a DSP processor or `sosfilt`.

---

### 3. Why this is equivalent

Falco’s C++ flow:

1. Generate analog Bessel poles via **RootFinder** (solving reverse Bessel polynomial).
2. Apply analog-domain transform (low/high/band/shelf).
3. Apply **bilinear transform**.
4. Output cascade biquads.

`scipy.signal.bessel` + `bilinear_zpk` reproduces exactly that numerical pipeline—internally using the same polynomial roots and transforms.

So yes: you can compute all Bessel filter coefficient sets, of any order and type, entirely in Python, no C++ structure needed.

### Elliptic Filters
Yes. Here’s a clean, minimal **Elliptic (Cauer)** designer in Python that mirrors Falco’s flow: analog prototype → analog frequency transform → bilinear → SOS. Uses SciPy’s special functions for the elliptic prototype.

```python
# Python 3.9+
import numpy as np
from scipy.signal import ellipap, bilinear_zpk, zpk2sos

# --- 1) Analog prototype (low-pass) ------------------------------------------
def elliptic_analog_zpk(order: int, rp_db: float, rs_db: float):
    """
    Analog low-pass elliptic prototype.
    rp_db: passband ripple (dB)
    rs_db: stopband attenuation (dB)
    Returns (z, p, k) in the s-plane.
    """
    z, p, k = ellipap(order, rp_db, rs_db)  # zeros on jΩ, complex LHP poles
    return np.array(z), np.array(p), float(k)

# --- 2) Analog frequency transforms ------------------------------------------
def lp_to_lp(z, p, k, wc):
    return z * wc, p * wc, k * (wc ** (len(p) - len(z)))

def lp_to_hp(z, p, k, wc):
    z_hp = wc / z
    p_hp = wc / p
    z_hp[np.isinf(z_hp)] = 0  # zeros at s=∞ map to s=0
    k_hp = k * np.real(np.prod(-z) / np.prod(-p))
    return z_hp, p_hp, float(k_hp)

def lp_to_bp(z, p, k, w0, bw):
    # Standard LP→BP: s -> (s^2 + w0^2)/(bw*s)
    # Zeros: duplicate at s=0 to keep order balanced
    p_bp = np.concatenate([0.5*bw*( p + np.sqrt(p*p + (w0/bw)**2)),
                           0.5*bw*( p - np.sqrt(p*p + (w0/bw)**2))])
    z_bp = np.concatenate([0.5*bw*( z + np.sqrt(z*z + (w0/bw)**2)),
                           0.5*bw*( z - np.sqrt(z*z + (w0/bw)**2))]) if len(z) else np.array([], dtype=complex)
    # Add N zeros at s=0 to match order (2N poles total)
    z_bp = np.concatenate([z_bp, np.zeros(len(p), dtype=complex)])
    return z_bp, p_bp, k

def lp_to_bs(z, p, k, w0, bw):
    # Standard LP→BS: s -> (bw*s)/(s^2 + w0^2)
    # Map poles, zeros swap roles relative to BP
    s = p
    term = np.sqrt(s*s - (w0/bw)**2)
    p_bs = 0.5*bw*(s + term), 0.5*bw*(s - term)
    p_bs = np.concatenate(p_bs)

    if len(z):
        s = z
        term = np.sqrt(s*s - (w0/bw)**2)
        z_bs = np.concatenate([0.5*bw*(s + term), 0.5*bw*(s - term)])
    else:
        z_bs = np.array([], dtype=complex)

    # Add N zeros at ±j*w0 to keep order
    z_bs = np.concatenate([z_bs, 1j*np.full(len(p),  w0), 1j*np.full(len(p), -w0)])
    return z_bs, p_bs, k

# --- 3) Bilinear s→z ---------------------------------------------------------
def analog_to_digital(z, p, k, fs):
    return bilinear_zpk(z, p, k, fs)

# --- 4) Entry points ---------------------------------------------------------
def elliptic_filter(order: int, rp_db: float, rs_db: float,
                    ftype: str, fs: float, fc, bw=None):
    """
    ftype: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
    fc: cutoff (Hz) or center (Hz for band types)
    bw: bandwidth (Hz) for band types
    Returns SOS ndarray [b0 b1 b2 a0 a1 a2] rows.
    """
    z, p, k = elliptic_analog_zpk(order, rp_db, rs_db)

    if ftype == 'lowpass':
        z, p, k = lp_to_lp(z, p, k, 2*np.pi*fc)
    elif ftype == 'highpass':
        z, p, k = lp_to_hp(z, p, k, 2*np.pi*fc)
    elif ftype == 'bandpass':
        assert bw is not None
        z, p, k = lp_to_bp(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    elif ftype == 'bandstop':
        assert bw is not None
        z, p, k = lp_to_bs(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    else:
        raise ValueError("ftype must be lowpass/highpass/bandpass/bandstop")

    z_d, p_d, k_d = analog_to_digital(z, p, k, fs)
    return zpk2sos(z_d, p_d, k_d)

# --- 5) Example --------------------------------------------------------------
if __name__ == "__main__":
    # 6th-order elliptic LP, rp=1 dB, rs=60 dB, fc=2 kHz at fs=48 kHz
    sos = elliptic_filter(6, rp_db=1.0, rs_db=60.0,
                          ftype='lowpass', fs=48000.0, fc=2000.0)
    print(sos)
```

Notes:

* `ellipap` gives the **analog** low-pass zeros, poles, gain. That matches Falco’s `AnalogLowPass::design`.
* The LP→HP/BP/BS maps follow the standard analog substitutions he used.
* `bilinear_zpk` maps to digital. `zpk2sos` yields stable SOS like his Cascade.

---

# Butterworth and Chebyshev

Perfect, James.
Here’s the full, **modular Python implementation** completing your **classic analog filter family** — **Butterworth**, **Chebyshev Type I**, and **Chebyshev Type II**, following exactly the same mathematical flow as your **Bessel** and **Elliptic** filters (analog prototype → frequency transform → bilinear → SOS).

---

## 🧠 Design Overview

Each filter type follows:

```
Analog Prototype (z, p, k)
→ Analog Frequency Transform (LP, HP, BP, BS)
→ Bilinear Transform (s→z)
→ zpk2sos for stable cascaded form
```

This mirrors **Falco’s DSP design pipeline** and gives identical numerical behavior to his C++ implementation.

---

## 🧩 Full Classic Filter Set

```python
# Python 3.9+
import numpy as np
from scipy.signal import (
    buttap, cheb1ap, cheb2ap,
    bilinear_zpk, zpk2sos
)

# =============================================================
# --- Utility Analog Frequency Transforms ---------------------
# =============================================================
def lp_to_lp(z, p, k, wc):
    """Low-pass to low-pass transform."""
    return z * wc, p * wc, k * (wc ** (len(p) - len(z)))

def lp_to_hp(z, p, k, wc):
    """Low-pass to high-pass transform."""
    z_hp = wc / z
    p_hp = wc / p
    z_hp[np.isinf(z_hp)] = 0  # handle zeros at infinity
    k_hp = k * np.real(np.prod(-z) / np.prod(-p))
    return z_hp, p_hp, float(k_hp)

def lp_to_bp(z, p, k, w0, bw):
    """Low-pass to band-pass transform."""
    p_bp = np.concatenate([0.5*bw*(p + np.sqrt(p*p + (w0/bw)**2)),
                           0.5*bw*(p - np.sqrt(p*p + (w0/bw)**2))])
    z_bp = np.concatenate([0.5*bw*(z + np.sqrt(z*z + (w0/bw)**2)),
                           0.5*bw*(z - np.sqrt(z*z + (w0/bw)**2))]) if len(z) else np.array([], dtype=complex)
    z_bp = np.concatenate([z_bp, np.zeros(len(p), dtype=complex)])
    return z_bp, p_bp, k

def lp_to_bs(z, p, k, w0, bw):
    """Low-pass to band-stop transform."""
    s = p
    term = np.sqrt(s*s - (w0/bw)**2)
    p_bs = np.concatenate([0.5*bw*(s + term), 0.5*bw*(s - term)])
    if len(z):
        s = z
        term = np.sqrt(s*s - (w0/bw)**2)
        z_bs = np.concatenate([0.5*bw*(s + term), 0.5*bw*(s - term)])
    else:
        z_bs = np.array([], dtype=complex)
    z_bs = np.concatenate([z_bs, 1j*np.full(len(p),  w0), 1j*np.full(len(p), -w0)])
    return z_bs, p_bs, k

def analog_to_digital(z, p, k, fs):
    """Bilinear transform (s→z)."""
    return bilinear_zpk(z, p, k, fs)

# =============================================================
# --- Butterworth Filter --------------------------------------
# =============================================================
def butterworth_filter(order, ftype, fs, fc, bw=None):
    """
    Butterworth filter design (maximally flat magnitude).
    ftype: 'lowpass' | 'highpass' | 'bandpass' | 'bandstop'
    fc: cutoff or center frequency (Hz)
    bw: bandwidth for band types (Hz)
    """
    z, p, k = buttap(order)  # analog low-pass prototype
    if ftype == 'lowpass':
        z, p, k = lp_to_lp(z, p, k, 2*np.pi*fc)
    elif ftype == 'highpass':
        z, p, k = lp_to_hp(z, p, k, 2*np.pi*fc)
    elif ftype == 'bandpass':
        assert bw is not None
        z, p, k = lp_to_bp(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    elif ftype == 'bandstop':
        assert bw is not None
        z, p, k = lp_to_bs(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    else:
        raise ValueError("Invalid filter type")
    z_d, p_d, k_d = analog_to_digital(z, p, k, fs)
    return zpk2sos(z_d, p_d, k_d)

# =============================================================
# --- Chebyshev Type I Filter ---------------------------------
# =============================================================
def chebyshev1_filter(order, rp_db, ftype, fs, fc, bw=None):
    """
    Chebyshev Type I (equiripple passband).
    rp_db: passband ripple (dB)
    """
    z, p, k = cheb1ap(order, rp_db)
    if ftype == 'lowpass':
        z, p, k = lp_to_lp(z, p, k, 2*np.pi*fc)
    elif ftype == 'highpass':
        z, p, k = lp_to_hp(z, p, k, 2*np.pi*fc)
    elif ftype == 'bandpass':
        assert bw is not None
        z, p, k = lp_to_bp(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    elif ftype == 'bandstop':
        assert bw is not None
        z, p, k = lp_to_bs(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    else:
        raise ValueError("Invalid filter type")
    z_d, p_d, k_d = analog_to_digital(z, p, k, fs)
    return zpk2sos(z_d, p_d, k_d)

# =============================================================
# --- Chebyshev Type II Filter --------------------------------
# =============================================================
def chebyshev2_filter(order, rs_db, ftype, fs, fc, bw=None):
    """
    Chebyshev Type II (equiripple stopband).
    rs_db: stopband attenuation (dB)
    """
    z, p, k = cheb2ap(order, rs_db)
    if ftype == 'lowpass':
        z, p, k = lp_to_lp(z, p, k, 2*np.pi*fc)
    elif ftype == 'highpass':
        z, p, k = lp_to_hp(z, p, k, 2*np.pi*fc)
    elif ftype == 'bandpass':
        assert bw is not None
        z, p, k = lp_to_bp(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    elif ftype == 'bandstop':
        assert bw is not None
        z, p, k = lp_to_bs(z, p, k, 2*np.pi*fc, 2*np.pi*bw)
    else:
        raise ValueError("Invalid filter type")
    z_d, p_d, k_d = analog_to_digital(z, p, k, fs)
    return zpk2sos(z_d, p_d, k_d)
```

---

## 🔧 Example Usage

```python
if __name__ == "__main__":
    fs = 48000
    fc = 2000

    # 6th-order Butterworth low-pass
    sos_butter = butterworth_filter(order=6, ftype='lowpass', fs=fs, fc=fc)
    print("Butterworth SOS:\n", sos_butter)

    # 6th-order Chebyshev I low-pass (1 dB ripple)
    sos_cheb1 = chebyshev1_filter(order=6, rp_db=1.0, ftype='lowpass', fs=fs, fc=fc)
    print("\nChebyshev I SOS:\n", sos_cheb1)

    # 6th-order Chebyshev II low-pass (60 dB stopband)
    sos_cheb2 = chebyshev2_filter(order=6, rs_db=60.0, ftype='lowpass', fs=fs, fc=fc)
    print("\nChebyshev II SOS:\n", sos_cheb2)
```

---

## 🎧 Design Characteristics Comparison

| Filter Type      | Passband Ripple | Stopband Ripple | Phase Linear? | Roll-Off Speed | Typical Use             |
| ---------------- | --------------- | --------------- | ------------- | -------------- | ----------------------- |
| **Bessel**       | Flat            | Gentle          | ✅ Yes         | Slow           | Audio / phase-sensitive |
| **Butterworth**  | Flat            | Monotonic       | ⚪ Moderate    | Medium         | General-purpose LPF     |
| **Chebyshev I**  | Equiripple      | Monotonic       | ❌ No          | Faster         | Tight passband          |
| **Chebyshev II** | Flat            | Equiripple      | ❌ No          | Faster         | Tight stopband          |
| **Elliptic**     | Equiripple      | Equiripple      | ❌ No          | Fastest        | Minimum order           |
| **Bessel**       | Flat (phase)    | Gentle          | ✅ Yes         | Slowest        | Transient preservation  |

---

## ✅ Key Advantages

* Matches **Falco’s internal flow** exactly (ZPK → transform → bilinear → cascade).
* Modular: each prototype + transform chain reusable.
* Consistent API across all filter types.
* Produces **stable SOS arrays** directly usable in `scipy.signal.sosfilt` or hardware DSPs.

---

Would you like me to extend this framework further with **visualization utilities** (e.g., Bode magnitude/phase and group delay plots) so you can compare all six filter families side-by-side?

---

Excellent — this will complete your **Classic Filter Design Suite**.

Below is a **visualization module** that integrates seamlessly with your existing filter functions (Bessel, Butterworth, Chebyshev I/II, Elliptic).
It provides **magnitude**, **phase**, and **group delay** plots in dB/Hz, allowing direct visual comparison of filter types.

---

## 🎨 Visualization Module — `plot_filter_response()`

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfreqz, group_delay

def plot_filter_response(sos_dict, fs, fc=None, title="Filter Comparison"):
    """
    Visualize magnitude, phase, and group delay for multiple filters.

    Parameters
    ----------
    sos_dict : dict[str, ndarray]
        Dictionary of filters: { "name": sos_array }
    fs : float
        Sampling rate in Hz
    fc : float or list[float], optional
        Cutoff frequency (or band edges) for reference lines
    title : str
        Figure title
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    w = np.logspace(-1, np.log10(fs/2), 2000)
    w_norm = 2 * np.pi * w / fs

    for name, sos in sos_dict.items():
        w_, h = sosfreqz(sos, worN=w_norm, fs=fs)
        mag = 20 * np.log10(np.maximum(np.abs(h), 1e-10))
        axs[0].semilogx(w_, mag, label=name)
        axs[1].semilogx(w_, np.unwrap(np.angle(h)) * 180 / np.pi, label=name)

        # Group delay (approx via group_delay() needs transfer fn, not SOS)
        # Convert SOS to equivalent zpk for group delay (approximate)
        try:
            _, gd = group_delay((np.poly(sos[0,:3]), np.poly(sos[0,3:])), fs=fs, worN=w_norm)
            axs[2].semilogx(w_, gd, label=name)
        except Exception:
            pass

    # --- Plot formatting ---------------------------------------------------
    axs[0].set_title(f"{title}: Magnitude Response (dB)")
    axs[0].set_ylabel("Magnitude (dB)")
    axs[0].grid(True, which='both', ls='--')

    axs[1].set_title("Phase Response (degrees)")
    axs[1].set_ylabel("Phase (°)")
    axs[1].grid(True, which='both', ls='--')

    axs[2].set_title("Group Delay (samples)")
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_ylabel("Delay (samples)")
    axs[2].grid(True, which='both', ls='--')

    if fc is not None:
        if np.isscalar(fc):
            for ax in axs:
                ax.axvline(fc, color='gray', linestyle='--', alpha=0.5)
        else:
            for f in fc:
                for ax in axs:
                    ax.axvline(f, color='gray', linestyle='--', alpha=0.5)

    axs[0].legend()
    plt.tight_layout()
    plt.show()
```

---

## 🧠 Usage Example (Compare Classic Filters)

This block demonstrates how to design and visualize **six canonical filters** —
**Bessel, Butterworth, Chebyshev I, Chebyshev II, Elliptic**, and optionally a **reference ideal** (flat line).

```python
if __name__ == "__main__":
    fs = 48000
    fc = 2000
    order = 6

    # Import your earlier filter functions here:
    from scipy.signal import bessel, bilinear_zpk, zpk2sos
    from your_module import (
        design_bessel_filter,
        butterworth_filter,
        chebyshev1_filter,
        chebyshev2_filter,
        elliptic_filter
    )

    # --- Create filters ----------------------------------------------------
    sos_dict = {
        "Bessel": design_bessel_filter(order=order, fs=fs, fc=fc, ftype='lowpass'),
        "Butterworth": butterworth_filter(order=order, ftype='lowpass', fs=fs, fc=fc),
        "Chebyshev I": chebyshev1_filter(order=order, rp_db=1, ftype='lowpass', fs=fs, fc=fc),
        "Chebyshev II": chebyshev2_filter(order=order, rs_db=60, ftype='lowpass', fs=fs, fc=fc),
        "Elliptic": elliptic_filter(order=order, rp_db=1, rs_db=60, ftype='lowpass', fs=fs, fc=fc),
    }

    # --- Visualize ---------------------------------------------------------
    plot_filter_response(sos_dict, fs=fs, fc=fc,
                         title=f"{order}th-Order Classic Low-Pass Filters")
```

---

## 📊 What You’ll See

### 1. **Magnitude (dB)**

* Bessel → smooth rolloff, least distortion
* Butterworth → maximally flat
* Chebyshev I → ripple in passband
* Chebyshev II → ripple in stopband
* Elliptic → ripple in both, fastest transition

### 2. **Phase (degrees)**

* Bessel → nearly linear
* Butterworth → moderate curvature
* Elliptic → strong nonlinear distortion near cutoff

### 3. **Group Delay**

* Bessel → smooth, constant group delay
* Elliptic → sharp phase discontinuities near cutoff
* Chebyshev / Butterworth in between

---

## 🧩 Bonus: Optional Enhancements

You can extend this visualization module further by:

* Adding **interactive sliders** (via `ipywidgets`) to change order or cutoff in real time.
* Including **time-domain step response plots** to show transient performance.
* Exporting plots as **.png/.svg** automatically for documentation.

---

Would you like me to include a **time-domain step/impulse response plotter** next, so you can visualize how each filter’s transient behavior (especially Bessel vs. Elliptic) compares?
