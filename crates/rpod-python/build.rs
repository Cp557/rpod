use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let rustc = env::var_os("RUSTC").expect("Cargo must provide RUSTC");
    let output = Command::new(rustc)
        .arg("-V")
        .output()
        .expect("could not run rustc for build provenance");
    assert!(output.status.success(), "rustc -V failed");
    let version = String::from_utf8(output.stdout).expect("rustc version must be UTF-8");
    println!("cargo:rustc-env=RPOD_RUSTC_VERSION={}", version.trim());
}
