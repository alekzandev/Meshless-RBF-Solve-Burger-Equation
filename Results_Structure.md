# Results Structure

```json
{
    "nu": "viscosity (float)",
    "RBF": "Radial basis function, TPS or MQ (string)",
    "points": {
        "Interior": "Interior collocation points (list)",
        "boundary": "Boundary collocation points (list)"
    },
    "solution": {
        "x.x": "Solution on interior collocation points at time x.x (list)"
    }
}
```