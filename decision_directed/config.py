from dataclasses import dataclass


@dataclass
class DecisionDirectedConfig:
    llr_threshold: float = 4.0
    normalizer_step_size: float = 0.01
    noise_step_size: float = 0.05
    min_acceptance_rate: float = 0.1
    max_acceptance_rate: float = 0.9
    adaptive_threshold: bool = True
    threshold_adapt_rate: float = 0.001
    use_soft_gating: bool = False
    soft_temperature: float = 1.0
    track_per_tone: bool = True
    enable_noise_tracking: bool = True
    initialize_noise_from_pilots: bool = True
    clip_normalizer: bool = True
    normalizer_clip_value: float = 3.0
    
    @property
    def decision_directed_enabled(self) -> bool:
        return True