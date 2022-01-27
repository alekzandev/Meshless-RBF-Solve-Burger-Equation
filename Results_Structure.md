# Results Structure

Files located on `data/simulations` with the following format:

`solution_<RBF>_Mi_<Interior>_Mb_<boundary>_nu_<nu>_<poly>.json`

```json
{
    "poly": "polynomial basis: arb, laguerre, hermite (str)",
    "nu": "viscosity (float)",
    "RBF": "Radial basis function: TPS or MQ (string)",
    "points": {
        "Interior": "Interior collocation points (list)",
        "boundary": "Boundary collocation points (list)"
    },
    "solution": {
        "x.x": "Solution on interior collocation points at time x.x (list)"
    }
}
```