import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import pickle
from tqdm import tqdm

# Qiskit imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.fake_provider import FakeQuito
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit.utils import algorithm_globals

# Dataset import
from torchquantum.dataset import MNIST


class Qiskit_Quantum_Model(torch.nn.Module):
    def __init__(self, n_qubits=4, use_noise_model=True, backend_device='CPU'):
        super().__init__()
        self.n_qubits = n_qubits
        self.use_noise_model = use_noise_model
        self.backend_device = backend_device

        # Trainable quantum circuit parameters
        self.u3_params = torch.nn.Parameter(torch.randn(n_qubits, 3) * 0.1, requires_grad=True)
        self.cu3_params = torch.nn.Parameter(torch.randn(n_qubits, 3) * 0.1, requires_grad=True)

        # Setup Qiskit noise backend
        self.setup_qiskit_noise_backend()

    def setup_qiskit_noise_backend(self):
        """Setup Qiskit noise backend"""
        if self.use_noise_model:
            try:
                with open('NoiseModel/my_fake_quito_noise.pkl', 'rb') as file:
                    noise_model_dict = pickle.load(file)
                self.noise_model = AerNoiseModel().from_dict(noise_model_dict)
                print("Successfully loaded custom noise model")
            except Exception as e:
                print(f"Error loading noise model: {e}")
                raise
        else:
            self.noise_model = None
            print("Running without noise model")

        self.shot = 6000
        self.seeds = 170
        algorithm_globals.random_seed = self.seeds

        # Use FakeQuito backend (5 qubits)
        self.system_model = FakeQuito()

    def create_quantum_circuit(self, x):
        # Preprocess data: downsample, flatten and scale
        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        x_scaled = x * np.pi

        quantum_circuits = []

        for i in range(bsz):
            qc = QuantumCircuit(self.n_qubits)

            # Part 1: data encoder
            gate_sequence = ['ry'] * self.n_qubits + ['rx'] * self.n_qubits + ['rz'] * self.n_qubits + [
                'ry'] * self.n_qubits

            for j, gate_type in enumerate(gate_sequence):
                qubit_idx = j % self.n_qubits
                angle = float(x_scaled[i, j].detach())

                if gate_type == 'ry':
                    qc.ry(angle, qubit_idx)
                elif gate_type == 'rx':
                    qc.rx(angle, qubit_idx)
                elif gate_type == 'rz':
                    qc.rz(angle, qubit_idx)

            # Part 2: U3 gates (with trainable parameters)
            for qubit in range(self.n_qubits):
                # Use detach() to avoid warnings
                # theta = float(self.u3_params[qubit, 0].detach())
                # phi = float(self.u3_params[qubit, 1].detach())
                # lam = float(self.u3_params[qubit, 2].detach())
                theta = float(self.u3_params[qubit, 0])
                phi = float(self.u3_params[qubit, 1])
                lam = float(self.u3_params[qubit, 2])
                qc.u(theta, phi, lam, qubit)

            # Part 3: CU3 gates (with trainable parameters)
            connections = [(i, (i + 1) % self.n_qubits) for i in range(self.n_qubits)]

            for idx, (control, target) in enumerate(connections):
                # Use detach() to avoid warnings
                # theta = float(self.cu3_params[idx, 0].detach())
                # phi = float(self.cu3_params[idx, 1].detach())
                # lam = float(self.cu3_params[idx, 2].detach())
                theta = float(self.cu3_params[idx, 0])
                phi = float(self.cu3_params[idx, 1])
                lam = float(self.cu3_params[idx, 2])
                qc.cu(theta, phi, lam, 0, control, target)

            quantum_circuits.append(qc)

        return quantum_circuits

    def adjust_observable_for_circuit(self, observable, circuit):
        """Adjust observable based on circuit qubit count"""
        circuit_qubits = circuit.num_qubits
        observable_qubits = len(observable.paulis[0])

        if circuit_qubits == observable_qubits:
            return observable
        elif circuit_qubits > observable_qubits:
            padding = "I" * (circuit_qubits - observable_qubits)
            pauli_strs = [str(pauli) for pauli in observable.paulis]
            adjusted_paulis = [pauli_str + padding for pauli_str in pauli_strs]
            return SparsePauliOp(adjusted_paulis)
        else:
            pauli_strs = [str(pauli) for pauli in observable.paulis]
            adjusted_paulis = [pauli_str[:circuit_qubits] for pauli_str in pauli_strs]
            return SparsePauliOp(adjusted_paulis)

    def run_qiskit_simulator(self, quantum_circuits, observable):
        results = []

        for i, qc in enumerate(quantum_circuits):
            try:
                transpiled_qc = transpile(qc, backend=self.system_model, optimization_level=1)
            except Exception as e:
                print(f"Error transpiling circuit {i}: {e}")
                transpiled_qc = qc

            try:
                adjusted_observable = self.adjust_observable_for_circuit(observable, transpiled_qc)

                backend_options = {
                    'method': 'statevector',
                    'device': self.backend_device,
                }
                if self.use_noise_model:
                    backend_options['noise_model'] = self.noise_model

                estimator = Estimator(
                    backend_options=backend_options,
                    run_options={
                        'shots': self.shot,
                        'seed': self.seeds,
                    },
                    skip_transpilation=True
                )

                job = estimator.run(transpiled_qc, adjusted_observable)
                result = job.result()
                expectation_value = result.values[0]
                results.append(expectation_value)

            except Exception as e:
                print(f"Error running quantum circuit {i}: {e}")
                results.append(0.0)

        return torch.tensor(results, dtype=torch.float32, requires_grad=True)


    def forward(self, x):
        device = x.device

        # Create quantum circuits
        quantum_circuits = self.create_quantum_circuit(x)

        # Create Pauli-Z observable
        observable = SparsePauliOp.from_list([
            ("ZIII", 1.0),
            ("IZII", 1.0),
            ("IIZI", 1.0),
            ("IIIZ", 1.0)
        ])

        # Run qiskit simulator
        quantum_results = self.run_qiskit_simulator(quantum_circuits, observable)
        quantum_results = quantum_results.to(device)

        # Ensure results require gradients
        if not quantum_results.requires_grad:
            quantum_results.requires_grad_(True)

        # Convert single expectation value to binary classification output
        # Method 1: Use sigmoid function to map expectation values to [0,1] range
        class0_prob = torch.sigmoid(quantum_results)
        class1_prob = 1 - class0_prob
        quantum_output = torch.stack([class0_prob, class1_prob], dim=1)

        # Method 2: Directly use expectation values as logits, let softmax handle it
        # quantum_output = torch.stack([quantum_results, -quantum_results], dim=1)

        # Apply log softmax
        output = F.log_softmax(quantum_output, dim=1)

        return output


def train(dataflow, model, device, optimizer):
    pbar = tqdm(dataflow["train"], desc="Training", unit="batch")

    total_loss = 0.0
    for batch_idx, feed_dict in enumerate(pbar):
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{avg_loss:.4f}'
        })


def valid_test(dataflow, split, model, device):
    target_all = []
    output_all = []

    with torch.no_grad():
        pbar = tqdm(dataflow[split], desc=f"{split.capitalize()}ing", unit="batch")
        for feed_dict in pbar:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs)

            target_all.append(targets)
            output_all.append(outputs)

        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy:.4f}")
    print(f"{split} set loss: {loss:.4f}")

    return accuracy, loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--no-noise", action="store_false", dest="use_noise_model",
                        help="disable noise model", default=False)  # Set 'True' if using backend noise
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")

    # Add new parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="device for torch (cuda/cpu)")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--backend-device", type=str, default="CPU", choices=["CPU", "GPU"],
                        help="device for Qiskit backend (CPU/GPU)")

    args = parser.parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    dataset = MNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=[3, 6],
    )

    dataflow = dict()
    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0,
            pin_memory=False,
        )

    device = torch.device(args.device)

    print(f"Training configuration:")
    print(f"Using device: {device}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Use noise model: {args.use_noise_model}")
    print(f"  Torch device: {args.device}")
    print(f"  Qiskit backend device: {args.backend_device}")

    model = Qiskit_Quantum_Model(
        n_qubits=4,
        use_noise_model=args.use_noise_model,
        backend_device=args.backend_device  # Pass backend device parameter
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    accuracy_list = []
    loss_list = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}:")

        # Training phase
        train(dataflow, model, device, optimizer)

        # Validation phase
        accuracy, loss = valid_test(dataflow, "valid", model, device)
        accuracy_list.append(accuracy)
        loss_list.append(loss)

        scheduler.step()

    # Testing phase
    print("\nFinal Testing:")
    valid_test(dataflow, "test", model, device)


if __name__ == "__main__":
    main()
