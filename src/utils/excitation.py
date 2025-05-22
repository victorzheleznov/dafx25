import torch


class Excitation:
    """Class for excitation."""
    def __init__(
            self,
            amp: torch.Tensor = None,
            dur: torch.Tensor = None,
            st: torch.Tensor = None,
            type: torch.Tensor = None
        ):
        """Parameters
        ----------
        amp : torch.Tensor
            Amplitudes (batch size,).
        dur : torch.Tensor
            Durations (batch size,).
        st : torch.Tensor
            Starting times (batch size,).
        type : torch.Tensor
            Types (batch size,): 1 - pluck, 2 - strike.
        """
        self._amp = amp
        self._dur = dur
        self._st = st
        self._type = type.int() if type is not None else type

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute excitation.

        Parameters
        ----------
        t : torch.Tensor
            Time vector (number of samples,)
        """
        if None not in (self._amp, self._dur, self._st, self._type):
            fe = 0.5 * self._amp.unsqueeze(-1) * (
                1.0 - torch.cos(self._type.unsqueeze(-1) * torch.pi * (t.unsqueeze(0) - self._st.unsqueeze(-1)) / self._dur.unsqueeze(-1))
            ) * torch.logical_and(
                (t.unsqueeze(0) >= self._st.unsqueeze(-1)),
                (t.unsqueeze(0) <= (self._st + self._dur).unsqueeze(-1))
            ).int()
        else:
            fe = torch.zeros((1, *t.shape), device=t.device, dtype=t.dtype)
        return fe

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, value: torch.Tensor):
        self._amp = value

    @property
    def dur(self):
        return self._dur

    @dur.setter
    def dur(self, value: torch.Tensor):
        self._dur = value

    @property
    def st(self):
        return self._st

    @st.setter
    def st(self, value: torch.Tensor):
        self._st = value

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value: torch.Tensor):
        self._type = value.int() if value is not None else value
