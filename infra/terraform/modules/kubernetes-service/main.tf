variable "model_name" {
  type = string
}

variable "image_uri" {
  type = string
}

variable "desired_replicas" {
  type = number
}

variable "min_replicas" {
  type = number
}

variable "max_replicas" {
  type = number
}

variable "canary_percent" {
  type = number
}

resource "terraform_data" "workload" {
  input = {
    deployment_name = var.model_name
    image_uri       = var.image_uri
    replicas        = var.desired_replicas
    autoscaling = {
      min = var.min_replicas
      max = var.max_replicas
    }
    mesh = {
      enabled       = true
      routing_split = var.canary_percent > 0 ? [100 - var.canary_percent, var.canary_percent] : [100, 0]
      failover      = true
    }
    probes = {
      readiness = "/healthz/ready"
      liveness  = "/healthz/live"
    }
  }
}

output "rollout_policy" {
  value = terraform_data.workload.input.mesh
}
