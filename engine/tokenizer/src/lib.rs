use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;
use std::sync::OnceLock;

use tokenizers::utils::parallelism;
use tokenizers::Tokenizer;

const ABI_VERSION: i32 = 1;
const ERR_NULL: i32 = -1;
const ERR_VERSION: i32 = -2;
const ERR_UTF8: i32 = -3;
const ERR_ENCODE: i32 = -4;

#[repr(C)]
pub struct inf_tok;

pub struct TokenizerWrapper {
    tokenizer: Tokenizer,
    buffer: Vec<i32>,
    chunk_size: usize,
    offset: usize,
    prefix_enabled: bool,
    prefix_k: usize,
    prefix_capacity: usize,
    cache_map: HashMap<String, Vec<i32>>,
    cache_order: VecDeque<String>,
    last_error: Option<CString>,
    max_threads: Option<usize>,
}

impl TokenizerWrapper {
    fn new(model_path: &str) -> Result<Self, String> {
        let tokenizer = Tokenizer::from_file(model_path).map_err(|e| e.to_string())?;
        Ok(Self {
            tokenizer,
            buffer: Vec::new(),
            chunk_size: 0,
            offset: 0,
            prefix_enabled: false,
            prefix_k: 128,
            prefix_capacity: 32,
            cache_map: HashMap::new(),
            cache_order: VecDeque::new(),
            last_error: None,
            max_threads: None,
        })
    }

    fn set_last_error(&mut self, msg: &str) -> i32 {
        self.last_error = Some(CString::new(msg).unwrap_or_else(|_| CString::new("tokenizer error").unwrap()));
        ERR_ENCODE
    }

    fn make_cache_key(&self, text: &str) -> String {
        if self.prefix_k == 0 {
            return String::new();
        }
        text.chars().take(self.prefix_k).collect()
    }

    fn get_cached(&mut self, key: &str) -> Option<Vec<i32>> {
        if let Some(value) = self.cache_map.get(key) {
            if let Some(pos) = self.cache_order.iter().position(|k| k == key) {
                self.cache_order.remove(pos);
            }
            self.cache_order.push_back(key.to_string());
            return Some(value.clone());
        }
        None
    }

    fn insert_cache(&mut self, key: String, value: Vec<i32>) {
        if self.prefix_capacity == 0 {
            return;
        }
        if self.cache_map.contains_key(&key) {
            self.cache_map.insert(key.clone(), value);
            if let Some(pos) = self.cache_order.iter().position(|k| k == &key) {
                self.cache_order.remove(pos);
            }
            self.cache_order.push_back(key);
            return;
        }
        if self.cache_map.len() >= self.prefix_capacity {
            if let Some(old_key) = self.cache_order.pop_front() {
                self.cache_map.remove(&old_key);
            }
        }
        self.cache_order.push_back(key.clone());
        self.cache_map.insert(key, value);
    }

    fn begin(&mut self, text: &str, n_threads: Option<usize>) -> Result<(), String> {
        self.configure_parallelism(n_threads.or(self.max_threads));

        let key = if self.prefix_enabled { Some(self.make_cache_key(text)) } else { None };
        if let Some(ref k) = key {
            if let Some(cached) = self.get_cached(k) {
                self.buffer = cached;
                self.offset = 0;
                self.chunk_size = self.buffer.len();
                return Ok(());
            }
        }

        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| e.to_string())?;
        self.buffer.clear();
        self.buffer.extend(encoding.get_ids().iter().map(|&id| id as i32));
        self.offset = 0;
        self.chunk_size = if let Some(threads) = n_threads.or(self.max_threads) {
            std::cmp::max(128, self.buffer.len() / threads.max(1))
        } else {
            self.buffer.len()
        };
        if self.chunk_size == 0 {
            self.chunk_size = self.buffer.len();
        }

        if let Some(k) = key {
            if self.prefix_enabled {
                self.insert_cache(k, self.buffer.clone());
            }
        }

        Ok(())
    }

    fn next(&mut self) -> Option<(*const i32, usize)> {
        if self.offset >= self.buffer.len() {
            return None;
        }
        let len = std::cmp::min(self.chunk_size, self.buffer.len() - self.offset);
        let ptr = unsafe { self.buffer.as_ptr().add(self.offset) };
        self.offset += len;
        Some((ptr, len))
    }

    fn end(&mut self) {
        self.buffer.clear();
        self.offset = 0;
        self.chunk_size = 0;
    }
}

impl TokenizerWrapper {
    fn configure_parallelism(&self, threads: Option<usize>) {
        static RAYON_THREADS: OnceLock<usize> = OnceLock::new();

        match threads {
            Some(t) if t > 1 => {
                parallelism::set_parallelism(true);
                if RAYON_THREADS.get().copied().unwrap_or(t) == t {
                    if RAYON_THREADS.set(t).is_ok() {
                        std::env::set_var("RAYON_NUM_THREADS", t.to_string());
                    }
                }
            }
            Some(_) => {
                parallelism::set_parallelism(false);
            }
            None => {
                parallelism::set_parallelism(true);
            }
        }
    }
}

#[repr(C)]
pub struct inf_tok_batch_t {
    pub ids: *const i32,
    pub len: usize,
}

#[no_mangle]
pub extern "C" fn inf_tok_new(abi_version: i32, model_path: *const c_char) -> *mut inf_tok {
    if model_path.is_null() {
        return ptr::null_mut();
    }
    if abi_version != ABI_VERSION {
        return ptr::null_mut();
    }
    let c_str = unsafe { CStr::from_ptr(model_path) };
    let model = match c_str.to_str() {
        Ok(s) => s,
        Err(_) => return ptr::null_mut(),
    };
    match TokenizerWrapper::new(model) {
        Ok(wrapper) => Box::into_raw(Box::new(wrapper)) as *mut inf_tok,
        Err(_) => ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn inf_tok_free(tokenizer: *mut inf_tok) {
    if tokenizer.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(tokenizer as *mut TokenizerWrapper));
    }
}

#[no_mangle]
pub extern "C" fn inf_tok_set_prefix_cache(tokenizer: *mut inf_tok, enable: i32) {
    if tokenizer.is_null() {
        return;
    }
    let wrapper = unsafe { &mut *(tokenizer as *mut TokenizerWrapper) };
    wrapper.prefix_enabled = enable != 0;
    if !wrapper.prefix_enabled {
        wrapper.cache_map.clear();
        wrapper.cache_order.clear();
    }
}

#[no_mangle]
pub extern "C" fn inf_tok_set_prefix_params(tokenizer: *mut inf_tok, prefix_k: usize, capacity: usize) {
    if tokenizer.is_null() {
        return;
    }
    let wrapper = unsafe { &mut *(tokenizer as *mut TokenizerWrapper) };
    wrapper.prefix_k = prefix_k;
    wrapper.prefix_capacity = capacity;
    if wrapper.cache_map.len() > wrapper.prefix_capacity {
        wrapper.cache_map.clear();
        wrapper.cache_order.clear();
    }
}

#[no_mangle]
pub extern "C" fn inf_tok_set_max_threads(tokenizer: *mut inf_tok, threads: usize) {
    if tokenizer.is_null() {
        return;
    }
    let wrapper = unsafe { &mut *(tokenizer as *mut TokenizerWrapper) };
    wrapper.max_threads = if threads == 0 { None } else { Some(threads) };
}

#[no_mangle]
pub extern "C" fn inf_tok_encode_stream_begin(tokenizer: *mut inf_tok, text: *const c_char, n_threads: usize) -> i32 {
    if tokenizer.is_null() || text.is_null() {
        return ERR_NULL;
    }
    let wrapper = unsafe { &mut *(tokenizer as *mut TokenizerWrapper) };
    let text_c = unsafe { CStr::from_ptr(text) };
    let text_str = match text_c.to_str() {
        Ok(s) => s,
        Err(_) => {
            wrapper.last_error = Some(CString::new("invalid UTF-8").unwrap());
            return ERR_UTF8;
        }
    };
    match wrapper.begin(text_str, if n_threads == 0 { wrapper.max_threads } else { Some(n_threads) }) {
        Ok(_) => 0,
        Err(err) => wrapper.set_last_error(&err),
    }
}

#[no_mangle]
pub extern "C" fn inf_tok_encode_stream_next(tokenizer: *mut inf_tok, out_batch: *mut inf_tok_batch_t) -> i32 {
    if tokenizer.is_null() || out_batch.is_null() {
        return ERR_NULL;
    }
    let wrapper = unsafe { &mut *(tokenizer as *mut TokenizerWrapper) };
    match wrapper.next() {
        Some((ptr, len)) => {
            unsafe {
                (*out_batch).ids = ptr;
                (*out_batch).len = len;
            }
            1
        }
        None => {
            unsafe {
                (*out_batch).ids = std::ptr::null();
                (*out_batch).len = 0;
            }
            0
        }
    }
}

#[no_mangle]
pub extern "C" fn inf_tok_encode_stream_end(tokenizer: *mut inf_tok) {
    if tokenizer.is_null() {
        return;
    }
    let wrapper = unsafe { &mut *(tokenizer as *mut TokenizerWrapper) };
    wrapper.end();
}

#[no_mangle]
pub extern "C" fn inf_tok_last_error(tokenizer: *const inf_tok) -> *const c_char {
    if tokenizer.is_null() {
        return ptr::null();
    }
    let wrapper = unsafe { &*(tokenizer as *const TokenizerWrapper) };
    match &wrapper.last_error {
        Some(err) => err.as_ptr(),
        None => ptr::null(),
    }
}
