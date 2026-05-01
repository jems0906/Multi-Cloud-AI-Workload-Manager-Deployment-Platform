from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
import time

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


from .api import FlexAIClient

app = typer.Typer(add_completion=False, help="Deploy and manage AI workloads across clouds.")
console = Console()


def ensure_dockerfile(artifact_path: Path) -> Path:
    dockerfile_path = artifact_path.parent / "Dockerfile.flexai"
    if dockerfile_path.exists():
        return dockerfile_path

    dockerfile_path.write_text(
        "FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04\n"
        "WORKDIR /app\n"
        "COPY . /app\n"
        "RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*\n"
        "RUN if [ -f requirements.txt ]; then pip3 install -r requirements.txt; fi\n"
        'CMD ["python3", "serve.py"]\n',
        encoding="utf-8",
    )
    return dockerfile_path


@app.command()
def deploy(
    artifact_path: Path,
    model_name: str = typer.Option(..., "--model-name"),
    gpu: str = typer.Option("A100", "--gpu"),
    region: str = typer.Option("us-west", "--region"),
    cloud: str = typer.Option("aws", "--cloud"),
    replicas: int = typer.Option(1, "--replicas"),
    runtime: str = typer.Option("custom", "--runtime"),
    canary_percent: int = typer.Option(0, "--canary-percent"),
    image_tag: str = typer.Option("", "--image-tag", help="Docker image tag (default: flexai/<model>:<id>)"),
    skip_docker: bool = typer.Option(False, "--skip-docker", help="Skip docker build/push (artifact already in registry)"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    dockerfile_path = ensure_dockerfile(artifact_path)
    deploy_id = uuid.uuid4().hex[:8]
    tag = image_tag or f"flexai/{model_name.lower().replace(' ', '-')}:{deploy_id}"

    if not skip_docker:
        console.print(f"[dim]Building Docker image {tag}...[/dim]")
        try:
            subprocess.run(
                ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(artifact_path.resolve().parent)],
                check=True,
            )
            console.print(f"[dim]Pushing image {tag}...[/dim]")
            subprocess.run(["docker", "push", tag], check=True)
        except FileNotFoundError:
            console.print("[yellow]docker not found on PATH — skipping build/push. Set --skip-docker to suppress this warning.[/yellow]")
        except subprocess.CalledProcessError as exc:
            raise typer.Exit(code=exc.returncode) from exc

    client = FlexAIClient(base_url=base_url)
    deployment = client.deploy(
        artifact_path,
        {
            "model_name": model_name,
            "runtime": runtime,
            "gpu": gpu,
            "region": region,
            "cloud": cloud,
            "replicas": replicas,
            "min_replicas": 1,
            "max_replicas": max(replicas * 4, 4),
            "canary_percent": canary_percent,
            "labels": {"source": "cli", "dockerfile": str(dockerfile_path.name), "image": tag},
        },
    )
    console.print(Panel.fit(f"Deployment created: {deployment['deployment_id']}\nEndpoint: {deployment['endpoint']}", title="FlexAI"))
    stream_logs(deployment["deployment_id"], base_url=base_url, iterations=3)


@app.command("list")
def list_deployments(base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url")) -> None:
    client = FlexAIClient(base_url=base_url)
    items = client.list_deployments()
    table = Table(title="Deployments")
    for column in ["ID", "Model", "Status", "Cloud", "Region", "GPU"]:
        table.add_column(column)
    for item in items:
        table.add_row(
            item["deployment_id"][:8],
            item["model_name"],
            item["status"],
            item["cloud"],
            item["region"],
            item["gpu"],
        )
    console.print(table)


@app.command()
def status(deployment_id: str, base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url")) -> None:
    client = FlexAIClient(base_url=base_url)
    deployment = client.get_deployment(deployment_id)
    console.print_json(json.dumps(deployment))


@app.command()
def logs(
    deployment_id: str,
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
    iterations: int = typer.Option(5, "--iterations"),
) -> None:
    stream_logs(deployment_id, base_url, iterations)


@app.command()
def rollback(
    deployment_id: str,
    target_version: str = typer.Option(..., "--target-version"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    client = FlexAIClient(base_url=base_url)
    deployment = client.rollback(deployment_id, target_version)
    console.print(Panel.fit(f"Active version: {deployment['active_version']}", title="Rollback complete"))


@app.command("ab-test")
def ab_test(
    deployment_id: str,
    challenger_version: str = typer.Option(..., "--version"),
    challenger_image_uri: str = typer.Option(..., "--image"),
    weight: int = typer.Option(10, "--weight"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    client = FlexAIClient(base_url=base_url)
    deployment = client.ab_test(deployment_id, challenger_version, challenger_image_uri, weight)
    console.print(Panel.fit(f"A/B test configured with {len(deployment['versions'])} versions", title="Traffic Split"))


@app.command()
def overview(base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url")) -> None:
    client = FlexAIClient(base_url=base_url)
    payload = client.overview()
    console.print_json(json.dumps(payload))


def stream_logs(deployment_id: str, base_url: str, iterations: int) -> None:
    """Stream live logs via WebSocket, falling back to REST polling."""
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_endpoint = f"{ws_url}/ws/deployments/{deployment_id}/logs"
    client = FlexAIClient(base_url=base_url)
    try:
        import websocket  # type: ignore[import]  # websocket-client

        count = 0
        console.print(f"[dim]Streaming logs from {ws_endpoint}[/dim]")

        def on_message(ws, message):  # noqa: ANN001
            nonlocal count
            console.print(message)
            count += 1
            if count >= iterations * 5:
                ws.close()

        def on_error(ws, error):  # noqa: ANN001
            console.print(f"[yellow]WebSocket error: {error}[/yellow]")

        ws_obj = websocket.WebSocketApp(
            f"{ws_endpoint}?token={client.token}",
            on_message=on_message,
            on_error=on_error,
        )
        ws_obj.run_forever()
    except ImportError:
        # websocket-client not installed — fall back to REST polling
        console.print("[dim]websocket-client not installed; polling REST instead[/dim]")
        for _ in range(iterations):
            deployment = client.get_deployment(deployment_id)
            log_tail = "\n".join(deployment["logs"][-5:])
            metrics = deployment["metrics"]
            console.print(
                Panel.fit(
                    f"{log_tail}\n\nGPU: {metrics['gpu_utilization']}% | "
                    f"Latency: {metrics['inference_latency_ms']} ms | "
                    f"Throughput: {metrics['throughput_rps']} rps",
                    title=f"{deployment['model_name']} \u00b7 {deployment['status']}",
                )
            )
            time.sleep(1)


@app.command()
def scale(
    deployment_id: str,
    replicas: int = typer.Option(..., "--replicas"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    """Scale a deployment to a given replica count."""
    client = FlexAIClient(base_url=base_url)
    deployment = client.scale(deployment_id, replicas)
    console.print(Panel.fit(f"Replicas: {deployment['replicas']}", title="Scaled"))


@app.command()
def delete(
    deployment_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    """Tear down a deployment and release all cloud resources."""
    if not yes:
        typer.confirm(f"Delete deployment {deployment_id}? This tears down all cloud resources.", abort=True)
    client = FlexAIClient(base_url=base_url)
    client.delete(deployment_id)
    console.print(Panel.fit(f"Deployment {deployment_id} deleted.", title="Deleted"))


@app.command()
def canary(
    deployment_id: str,
    percent: int = typer.Option(..., "--percent", help="Canary traffic percentage (0-100)"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    """Adjust live canary traffic split for a deployment."""
    client = FlexAIClient(base_url=base_url)
    deployment = client.update_canary(deployment_id, percent)
    console.print(
        Panel.fit(
            f"Canary: {deployment['canary_percent']}% | Strategy: {deployment['routing_strategy']}",
            title="Traffic Split Updated",
        )
    )


@app.command()
def mesh(
    deployment_id: str,
    enable: bool = typer.Option(..., "--enable/--disable", help="Enable or disable service mesh"),
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    """Enable or disable service mesh (mTLS, sidecar injection) for a deployment."""
    client = FlexAIClient(base_url=base_url)
    deployment = client.update_mesh(deployment_id, enable)
    state = "enabled" if deployment["mesh_enabled"] else "disabled"
    console.print(Panel.fit(f"Service mesh {state} for {deployment_id}", title="Mesh Updated"))


@app.command()
def budget(base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url")) -> None:
    """Show current hourly spend vs configured budget limits."""
    client = FlexAIClient(base_url=base_url)
    status = client.budget_status()
    table = Table(title="Budget Status")
    table.add_column("Scope")
    table.add_column("Current ($/hr)")
    table.add_column("Limit ($/hr)")
    table.add_column("Used %")
    global_limit = status.get("global_limit")
    global_current = status.get("global_current", 0.0)
    global_pct = status.get("global_pct")
    table.add_row(
        "GLOBAL",
        f"{global_current:.2f}",
        str(global_limit) if global_limit is not None else "unlimited",
        f"{global_pct:.1f}%" if global_pct is not None else "n/a",
    )
    for cloud, current in (status.get("cloud_current") or {}).items():
        limit = (status.get("cloud_limits") or {}).get(cloud)
        console.print(table)
        table.add_row(
            cloud.upper(),
            f"{current:.2f}",
            str(limit) if limit is not None else "unlimited",
            "n/a",
        )
    console.print(table)


@app.command("promote-failover")
def promote_failover(
    deployment_id: str,
    base_url: str = typer.Option("http://127.0.0.1:8000", "--api-url"),
) -> None:
    """Manually promote a healthy standby to primary for this deployment's failover group."""
    client = FlexAIClient(base_url=base_url)
    result = client.promote_failover(deployment_id)
    primary = result.get("primary", {})
    standbys = result.get("standbys", [])
    console.print(
        Panel.fit(
            f"New primary: {primary.get('deployment_id', '')[:8]} ({primary.get('region', '')})\n"
            f"Standbys: {len(standbys)}",
            title="Failover Promoted",
        )
    )


if __name__ == "__main__":
    app()
