output "deployment_summary" {
  value = {
    model_name   = var.model_name
    cloud        = var.cloud
    region       = var.region
    image_uri    = var.image_uri
    service_mesh = true
    observability = terraform_data.observability_stack.input
    security      = terraform_data.security_controls.input
    rollout       = module.kubernetes_service.rollout_policy
  }
}

output "registry_name" {
  value = terraform_data.container_registry.input.name
}
