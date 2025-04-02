import numpy as np

from strat_mode import StratBox
from sbox_param import ShearingBoxParam, ModeParam
import tracker as track
import finder as find

def equal_within_tolerance(x1, x2, tol=1.0e-12):
    if np.abs(x1 - x2) < tol:
        return True
    return False

lin_a = ShearingBoxParam()
lin_a.param['metallicity'] = 0.03
lin_a.param['stokes_range'] = [1.0e-8, 0.01]
lin_a.param['viscous_alpha'] = 1.0e-6

def test_mode_finder():
    mode = ModeParam()
    mode.param['wave_number_x'] = 600
    mode.param['n_coll'] = 100
    mode.param['scale_l'] = 0.004

    sb = StratBox(lin_a)
    mf = find.ModeFinder(sb)

    e_val, e_vec, centres, radii = \
      mf.find_growing_at_real_part(real_part=-0.05,
                                   imag_range=[0,1.0],
                                   mode_param=mode,
                                   n_eig=20,
                                   flip_real_imag_flag=False)
    print('Eigenvalues found: ', e_val)

    assert len(e_val) == 3
    assert(equal_within_tolerance(e_val[0], -0.05911051+0.60220487j, tol=1.0e-3))
    assert(equal_within_tolerance(e_val[1], -0.03651012+0.510676j, tol=1.0e-3))
    assert(equal_within_tolerance(e_val[2], -0.0177798 +0.42165161j, tol=1.0e-3))

def test_single():
    mode = ModeParam()
    mode.param['wave_number_x'] = 600
    mode.param['n_coll'] = 100
    mode.param['scale_l'] = 0.004

    sb = StratBox(lin_a)

    eig, vec, rad = \
        sb.find_eigenvalues(mode,
                            sigma=-5.91105552e-02+0.60220491j,
                            n_eig=1)

    assert(equal_within_tolerance(eig[0], -5.91105552e-02+0.60220491j, tol=1.0e-3))

def test_tracker():
    mode = ModeParam()
    mode.param['wave_number_x'] = 600
    mode.param['n_coll'] = 100
    mode.param['scale_l'] = 0.004

    sb = StratBox(lin_a)

    eig, vec, rad = \
        sb.find_eigenvalues(mode,
                            sigma=-5.91105552e-02+0.60220491j,
                            n_eig=1)

    w = track.WaveNumberTracker(sb, filename=None)
    kx_plot = np.logspace(np.log10(600), np.log10(700), 10).astype(int)
    ev = w.track(kx_plot, eig[0], mode, maxn_coll=1200, label='VSI')
    print(ev)
    assert(equal_within_tolerance(ev[0][-1], -0.09003228+0.67918931j, tol=1.0e-3))
