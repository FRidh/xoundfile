{ buildPythonPackage

, black
, flit
, sphinx
, pylint

, pytest

, dask
, soundfile
, xarray

, notebook

}:

let
  # Development tools used during package build
  nativeBuildInputs = [
    black
    flit
    sphinx
    pylint
  ];

  # Run-time Python dependencies
  propagatedBuildInputs = [
    dask
    soundfile
    xarray
  ];

  # Test-dependencies
  checkInputs = [
    pytest
  ];

  # Other dev dependencies not used during build
  devInputs = [
    notebook
  ];

  allInputs = nativeBuildInputs ++ propagatedBuildInputs ++ checkInputs ++ devInputs;

  pkg = buildPythonPackage {
    pname = "xoundfile";
    version = "dev";
    format = "pyproject";

    src = ./.;

    inherit nativeBuildInputs propagatedBuildInputs checkInputs;

    preBuild = ''
      echo "Checking for errors with pylint..."
      pylint -E xoundfile
    '';

    postInstall = ''
      echo "Checking formatting..."
      black --check xoundfile

      echo "Creating html docs..."
      make -C doc html
      mkdir -p $out
      mv doc/build/html $doc
    '';

    checkPhase = ''
      pytest tests
    '';

    doCheck = true;

    shellHook = ''
      export PYTHONPATH="$(pwd):"$PYTHONPATH""
    '';

    outputs = [
      "out"
      "doc"
    ];

    passthru = {
      inherit allInputs;
    };
  };

in pkg
