# Infrastructure

Terraform/OpenTofu configuration for provisioning AWS EC2 instances for vLLM experiments.

## Resources

- EC2 instance with configurable AMI and instance type
- Security group with SSH ingress (IPv4/IPv6) and unrestricted egress
- Auto-generated resource naming: `{org}-{owner}-{date}-{project}-{experiment}`
- Resources tagged with owner, project, experiment, and a 14-day TTL

## Usage

```sh
tofu init
tofu plan -var-file=dev.tfvars
tofu apply -var-file=dev.tfvars
```

## Variables

| Variable | Description | Default |
|---|---|---|
| `organization` | Organization name | `octo` |
| `owner` | Resource owner | |
| `project` | Project name | |
| `experiment` | Experiment identifier | |
| `ami` | EC2 AMI ID | |
| `instance_type` | EC2 instance type | |
| `key_name` | SSH key pair name | |

See `dev.tfvars` for current values.
