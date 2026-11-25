//! The JSON script layer of `chimera-serve` as a library: what a request to `/api/speak` may say
//! and the render it becomes ([`script`]), for the server (`main.rs`) and for `chimera-cli`.
//! Nothing here needs the HTTP stack: that is the binary's, behind the `server` feature.

pub mod script;
