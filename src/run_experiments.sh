# Burgers'

## baseline experiments
### fully observable
python sindyrl_TRAIN.py --baseline --filename "fom_burgers_FO.yml"
### partially observable
python sindyrl_TRAIN.py --baseline --filename "fom_burgers_PO.yml"

## auto encoder experiments
### fully observable
python sindyrl_TRAIN.py --no-baseline --filename "dyna_burgers_AE_FO_5.yml"
python sindyrl_TRAIN.py --no-baseline --filename "dyna_burgers_AE_FO_10.yml"

### partially observable
python sindyrl_TRAIN.py --no-baseline --filename "dyna_burgers_AE_PO_5.yml"
python sindyrl_TRAIN.py --no-baseline --filename "dyna_burgers_AE_PO_10.yml"

# Navier Stokes

## baseline
python sindyrl_TRAIN.py --baseline --filename "fom_navierStokes.yml"
## autoencoder
python sindyrl_TRAIN.py --no-baseline --filename "dyna_navierStokes_AE_5.yml"
python sindyrl_TRAIN.py --no-baseline --filename "dyna_navierStokes_AE_5_latentSmall.yml"