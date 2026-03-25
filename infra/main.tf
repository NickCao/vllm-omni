terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "6.28.0"
    }
    time = {
      source  = "hashicorp/time"
      version = "0.13.1"
    }
  }
}

variable "organization" {
  description = "Organization"
  type        = string
  default     = "octo"
}

variable "owner" {
  description = "Owner"
  type        = string
}

variable "project" {
  description = "Project"
  type        = string
}

variable "experiment" {
  description = "Experiment"
  type        = string
}

variable "ami" {
  description = "EC2 AMI"
  type        = string
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
}

variable "key_name" {
  description = "SSH key pair name"
  type        = string
}

resource "time_static" "created" {}

locals {
  timestamp = formatdate("YYYYMMDD", time_static.created.rfc3339)
  name      = "${var.organization}-${var.owner}-${local.timestamp}-${var.project}-${var.experiment}"
  tags = {
    Name          = local.name
    Organization  = var.organization
    Owner         = var.owner
    Timestamp     = local.timestamp
    Project       = var.project
    Experiment    = var.experiment
    TTL           = "14d"
    CleanupPolicy = "auto"
  }
}

provider "aws" {}

resource "aws_instance" "main" {
  ami           = var.ami
  instance_type = var.instance_type

  key_name = var.key_name

  vpc_security_group_ids = [
    aws_security_group.default.id
  ]

  tags = local.tags
}

resource "aws_security_group" "default" {
  name = local.name

  tags = local.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "default-ipv4" {
  security_group_id = aws_security_group.default.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "tcp"

  from_port = 22
  to_port   = 22

  description = "IPv4 allow SSH inbound"

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "default-ipv6" {
  security_group_id = aws_security_group.default.id

  cidr_ipv6   = "::/0"
  ip_protocol = "tcp"

  from_port = 22
  to_port   = 22

  description = "IPv6 allow SSH inbound"

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "http-ipv4" {
  security_group_id = aws_security_group.default.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "tcp"

  from_port = 80
  to_port   = 80

  description = "IPv4 allow HTTP inbound"

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "http-ipv6" {
  security_group_id = aws_security_group.default.id

  cidr_ipv6   = "::/0"
  ip_protocol = "tcp"

  from_port = 80
  to_port   = 80

  description = "IPv6 allow HTTP inbound"

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "https-ipv4" {
  security_group_id = aws_security_group.default.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "tcp"

  from_port = 443
  to_port   = 443

  description = "IPv4 allow HTTPS inbound"

  tags = local.tags
}

resource "aws_vpc_security_group_ingress_rule" "https-ipv6" {
  security_group_id = aws_security_group.default.id

  cidr_ipv6   = "::/0"
  ip_protocol = "tcp"

  from_port = 443
  to_port   = 443

  description = "IPv6 allow HTTPS inbound"

  tags = local.tags
}

resource "aws_vpc_security_group_egress_rule" "default-ipv4" {
  security_group_id = aws_security_group.default.id

  cidr_ipv4   = "0.0.0.0/0"
  ip_protocol = "-1"

  description = "IPv4 allow all outbound"

  tags = local.tags
}

resource "aws_vpc_security_group_egress_rule" "default-ipv6" {
  security_group_id = aws_security_group.default.id

  cidr_ipv6   = "::/0"
  ip_protocol = "-1"

  description = "IPv6 allow all outbound"

  tags = local.tags
}

output "instance_public_ip" {
  description = "Public IP of the EC2 instance"
  value       = aws_instance.main.public_ip
}
