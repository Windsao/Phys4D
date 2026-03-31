




import torch


class LinearInterpolation:
    """Linearly interpolates a sampled scalar function for arbitrary query points.

    This class implements a linear interpolation for a scalar function. The function maps from real values, x, to
    real values, y. It expects a set of samples from the function's domain, x, and the corresponding values, y.
    The class allows querying the function's values at any arbitrary point.

    The interpolation is done by finding the two closest points in x to the query point and then linearly
    interpolating between the corresponding y values. For the query points that are outside the input points,
    the class does a zero-order-hold extrapolation based on the boundary values. This means that the class
    returns the value of the closest point in x.
    """

    def __init__(self, x: torch.Tensor, y: torch.Tensor, device: str):
        """Initializes the linear interpolation.

        The scalar function maps from real values, x, to real values, y. The input to the class is a set of samples
        from the function's domain, x, and the corresponding values, y.

        Note:
            The input tensor x should be sorted in ascending order.

        Args:
            x: An vector of samples from the function's domain. The values should be sorted in ascending order.
                Shape is (num_samples,)
            y: The function's values associated to the input x. Shape is (num_samples,)
            device: The device used for processing.

        Raises:
            ValueError: If the input tensors are empty or have different sizes.
            ValueError: If the input tensor x is not sorted in ascending order.
        """

        self._x = x.view(-1).clone().to(device=device)
        self._y = y.view(-1).clone().to(device=device)


        if self._x.numel() == 0:
            raise ValueError("Input tensor x is empty!")
        if self._x.numel() != self._y.numel():
            raise ValueError(f"Input tensors x and y have different sizes: {self._x.numel()} != {self._y.numel()}")

        if torch.any(self._x[1:] < self._x[:-1]):
            raise ValueError("Input tensor x is not sorted in ascending order!")

    def compute(self, q: torch.Tensor) -> torch.Tensor:
        """Calculates a linearly interpolated values for the query points.

        Args:
           q: The query points. It can have any arbitrary shape.

        Returns:
            The interpolated values at query points. It has the same shape as the input tensor.
        """

        q_1d = q.view(-1)

        num_smaller_elements = torch.sum(self._x.unsqueeze(1) < q_1d.unsqueeze(0), dim=0, dtype=torch.int)



        lower_bound = torch.clamp(num_smaller_elements - 1, min=0)


        upper_bound = torch.clamp(num_smaller_elements, max=self._x.numel() - 1)


        weight = (q_1d - self._x[lower_bound]) / (self._x[upper_bound] - self._x[lower_bound])

        weight[upper_bound == lower_bound] = 0.0


        fq = self._y[lower_bound] + weight * (self._y[upper_bound] - self._y[lower_bound])


        fq = fq.view(q.shape)
        return fq
