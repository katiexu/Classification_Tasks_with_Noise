from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeKolkata
from qiskit.providers.fake_provider import FakeQuito

import pickle


# Get FakeQuito backend: 5 qubits
backend_fakequito = FakeQuito()
# # Get FakeKolkata backend: 27 qubits
# backend_fakekolkata = FakeKolkata()

# Extract noise model
noise_model = backend_fakequito.configuration().to_dict()

# Save to .pkl file
with open("NoiseModel/my_fake_quito_noise.pkl", "wb") as f:
    pickle.dump(noise_model, f)
