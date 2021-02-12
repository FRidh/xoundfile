{
  description = "Python module for loading sound files as xarray arrays.";

  inputs.nixpkgs.url = "nixpkgs/nixpkgs-unstable";
  inputs.utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, utils }: {
    overlay = final: prev: {
      pythonOverlay = pself: psuper: {
        xoundfile = pself.callPackage ./default.nix { };
      };
    };
  } // (utils.lib.eachSystem [ "x86_64-linux" ] (system: let
    # Our own overlay does not get applied to nixpkgs because that would lead to
    # an infinite recursion. Therefore, we need to import nixpkgs and apply it ourselves.
    pkgs = import nixpkgs {
      inherit system;
      overlays = [
          self.overlay
      ];
    };
    python = let
      mypython = pkgs.python3.override {
        packageOverrides = pkgs.pythonOverlay;
        self = mypython;
      };
    in mypython; 
  in rec {
    packages = rec {
      # Development environment that includes our package, its dependencies
      # and additional dev inputs.
      devEnv = python.withPackages(_: pkg.allInputs);
      pkg = python.pkgs.xoundfile;
    };

    defaultPackage = python.pkgs.xoundfile;
    devShell = pkgs.mkShell {
      nativeBuildInputs = [
        packages.devEnv
      ];
      shellHook = ''
        export PYTHONPATH=$(readlink -f .):$PYTHONPATH
      '';
    };
  }));
}
