import os
import subprocess
import tempfile

class MermaidGenerator:
    def __init__(self, config):
        self.executable = config["mermaid"]["executable_path"]

    def render(self, mermaid_code, output_path):
        """Renders Mermaid code to a PNG file using mermaid-cli."""
        
        # Create a temporary file for the mermaid code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mmd', delete=False, encoding='utf-8') as tmp:
            tmp.write(mermaid_code)
            tmp_path = tmp.name

        try:
            # Run mmdc (Mermaid CLI)
            # npx -y @mermaid-js/mermaid-cli -i input.mmd -o output.png
            # Check for local mmdc
            local_mmdc = os.path.join(os.getcwd(), "node_modules", ".bin", "mmdc")
            if os.path.exists(local_mmdc):
                cmd = [
                    local_mmdc,
                    "-i", tmp_path,
                    "-o", output_path,
                    "-b", "transparent"
                ]
            else:
                # Fallback to npx
                cmd = [
                    self.executable, "-y", "@mermaid-js/mermaid-cli",
                    "-i", tmp_path,
                    "-o", output_path,
                    "-b", "transparent"
                ]
            
            # On Windows, shell=True is often needed for npx to resolve correctly
            # Added timeout to prevent hanging
            result = subprocess.run(
                cmd, 
                check=False, 
                shell=False, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                timeout=120 # 120 seconds timeout
            )
            
            if result.returncode != 0:
                stderr_output = result.stderr.decode('utf-8', errors='replace')
                stdout_output = result.stdout.decode('utf-8', errors='replace')
                raise RuntimeError(f"Mermaid rendering failed (code {result.returncode}):\nSTDERR: {stderr_output}\nSTDOUT: {stdout_output}")
            
            if not os.path.exists(output_path):
                raise RuntimeError("Mermaid CLI finished successfully but output file was not created.")
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Mermaid rendering failed: {e.stderr.decode('utf-8')}")
        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
