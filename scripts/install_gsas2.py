"""Automated standalone installer and verifier for GSAS-II calculation engine."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from loguru import logger

GSAS2_REPO = "https://github.com/AdvancedPhotonSource/GSAS-II.git"
DEFAULT_INSTALL_DIR = Path.home() / ".GSASII" / "GSAS-II"


def install_gsas2(target_dir: Path | None = None, force: bool = False) -> bool:
    """Download and set up GSAS-II calculation engine with precompiled binaries."""
    install_path = target_dir or DEFAULT_INSTALL_DIR
    install_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Clone repository if not present
    if not (install_path / "GSASII").exists() or force:
        logger.info("Cloning GSAS-II repository into {}", install_path)
        cmd = ["git", "clone", "--depth", "1", GSAS2_REPO, str(install_path)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.debug("Git clone output: {}", res.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error("Git clone failed: {}", e)
            return False

    # 2. Add to sys.path
    install_str = str(install_path.resolve())
    if install_str not in sys.path:
        sys.path.insert(0, install_str)

    # 3. Download official pre-compiled APS binaries
    bindist_dir = install_path / "GSASII" / "bindist"
    bindist_dir.mkdir(parents=True, exist_ok=True)

    try:
        from GSASII import GSASIIpath

        loc = GSASIIpath.getGitBinaryLoc(verbose=True)
        if loc:
            logger.info("Downloading official pre-compiled APS binaries from {}", loc)
            GSASIIpath.InstallGitBinary(loc, str(bindist_dir), verbose=True)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as e:
        logger.warning("Could not auto-download binaries via GSASIIpath: {}", e)

    # 4. Windows MinGW DLL resolution
    if sys.platform == "win32":
        git_bin = shutil.which("git")
        if git_bin:
            git_mingw = Path(git_bin).resolve().parent.parent / "mingw64" / "bin"
            gcc_dll = git_mingw / "libgcc_s_seh-1.dll"
            if gcc_dll.exists():
                dest_dll = bindist_dir / "libgcc_s_seh-1.dll"
                if not dest_dll.exists():
                    shutil.copy2(gcc_dll, dest_dll)
                    logger.info("Co-located MinGW runtime DLL into bindist: {}", dest_dll)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(bindist_dir.resolve()))
            except OSError:
                pass

    # 5. Verify functional initialization
    try:
        from GSASII import GSASIIpath

        GSASIIpath.SetBinaryPath()
        if GSASIIpath.BinaryPathFailed:
            logger.warning("GSAS-II binary check reported BinaryPathFailed=True.")
            return False

        from GSASII import GSASIIscriptable as G2sc

        ver = G2sc.GetVersionNumber() if hasattr(G2sc, "GetVersionNumber") else "ready"
        logger.success("GSAS-II engine successfully installed and verified! Version: {}", ver)
        return True
    except (OSError, RuntimeError, ImportError, AttributeError, TypeError, ValueError) as e:
        logger.error("GSAS-II verification failed: {}", e)
        return False


if __name__ == "__main__":
    success = install_gsas2()
    sys.exit(0 if success else 1)
