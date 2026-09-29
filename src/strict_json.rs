//! Reject duplicate object keys, including criteria IDs, instead of silently replacing options.
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;

pub struct Strict(pub Value);

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StrictVisitor;
        impl<'de> Visitor<'de> for StrictVisitor {
            type Value = Strict;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value without duplicate keys")
            }
            fn visit_bool<E: de::Error>(self, v: bool) -> Result<Strict, E> {
                Ok(Strict(Value::Bool(v)))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<Strict, E> {
                Ok(Strict(Value::from(v)))
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<Strict, E> {
                Ok(Strict(Value::from(v)))
            }
            fn visit_f64<E: de::Error>(self, v: f64) -> Result<Strict, E> {
                Ok(Strict(Value::from(v)))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Strict, E> {
                Ok(Strict(Value::String(v.into())))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Strict, E> {
                Ok(Strict(Value::String(v)))
            }
            fn visit_none<E: de::Error>(self) -> Result<Strict, E> {
                Ok(Strict(Value::Null))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Strict, E> {
                Ok(Strict(Value::Null))
            }
            fn visit_some<D: Deserializer<'de>>(self, d: D) -> Result<Strict, D::Error> {
                Strict::deserialize(d)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Strict, A::Error> {
                let mut values = Vec::new();
                while let Some(Strict(value)) = seq.next_element::<Strict>()? {
                    values.push(value);
                }
                Ok(Strict(Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Strict, A::Error> {
                let mut values = Map::new();
                while let Some((key, Strict(value))) = map.next_entry::<String, Strict>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate key {key:?}")));
                    }
                    values.insert(key, value);
                }
                Ok(Strict(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(StrictVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nested_duplicates() {
        assert!(serde_json::from_str::<Strict>(r#"{"criteria":{"a":1,"a":2}}"#).is_err());
        assert!(serde_json::from_str::<Strict>(r#"{"criteria":{"a":1,"b":2}}"#).is_ok());
    }
}
