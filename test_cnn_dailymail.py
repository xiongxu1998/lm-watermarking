import os
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial

import torch
import random

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from extended_watermark_processor import WatermarkLogitsProcessor, WatermarkDetector

from demo_watermark import parse_args, load_model

from datasets import load_dataset
from evaluate import load


term_width = 80

def generate(prompt, args, model=None, device=None, tokenizer=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """

    watermark_processor = WatermarkLogitsProcessor(vocab=list(tokenizer.get_vocab().values()),
                                                    gamma=0.25,
                                                    delta=1.0,
                                                    seeding_scheme="selfhash",
                                                    select_green_tokens=args.select_green_tokens)

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, forced_eos_token_id=None)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True, 
            top_k=0,
            temperature=args.sampling_temp,
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams,
        ))

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]), 
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=args.prompt_max_length).to(device)
    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately: 
        torch.manual_seed(args.generation_seed)
    output_with_watermark = generate_with_watermark(**tokd_input)

    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:,tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = output_with_watermark[:,tokd_input["input_ids"].shape[-1]:]

    print("output_with_watermark:", output_with_watermark)

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    return (redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark, 
            decoded_output_with_watermark,
            args) 

def format_names(s):
    """Format names for the gradio demo interface"""
    s=s.replace("num_tokens_scored","Tokens Counted (T)")
    s=s.replace("num_green_tokens","# Tokens in Greenlist")
    s=s.replace("green_fraction","Fraction of T in Greenlist")
    s=s.replace("z_score","z-score")
    s=s.replace("p_value","p value")
    s=s.replace("prediction","Prediction")
    s=s.replace("confidence","Confidence")
    return s

def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k,v in score_dict.items():
        if k=='green_fraction': 
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k=='confidence': 
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float): 
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else: 
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2,["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1,["z-score Threshold", f"{detection_threshold}"])
    return lst_2d

def detect(input_text, args, model, device=None, tokenizer=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector(vocab=list(tokenizer.get_vocab().values()),
                                        gamma=0.25, # should match original setting
                                        seeding_scheme="selfhash", # should match original setting
                                        device=model.device, # must match the original rng device type
                                        tokenizer=tokenizer,
                                        z_threshold=4.0,
                                        normalizers=[],
                                        ignore_repeated_ngrams=True)
    
    if len(input_text)-1 > 4:
        score_dict = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = None
    return output, args


def test_cnn_dailymail(args):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])

    if not args.skip_model_load:
        model, tokenizer, device = load_model(args)
    else:
        model, tokenizer, device = None, None, None

    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
    perplexity = load("perplexity", module_type="metric")
    
    test_articles = dataset['test']['article']
    test_summaries = dataset['test']['highlights']
    num_type1_error = 0
    num_type2_error = 0
    all_watermarked_outputs = []
    all_unwatermarked_outputs = []
    
    random_indices = random.sample(range(len(test_articles)), 100)
    sampled_articles = [test_articles[i] for i in random_indices]
    
    invaild_inputs_num = 0
    
    if not args.skip_model_load:
        
        for article in sampled_articles:
            input_text = article

            args.default_prompt = input_text
            
            _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text, 
                                                                                            args, 
                                                                                            model=model, 
                                                                                            device=device, 
                                                                                            tokenizer=tokenizer)
            
            print("decoded_output_with_watermark: ", decoded_output_without_watermark)
            print("decoded_output_with_watermark: ", decoded_output_with_watermark)
            
            without_watermark_detection_result = detect(decoded_output_without_watermark, 
                                                    args,
                                                    model,
                                                    device=device, 
                                                    tokenizer=tokenizer)
            with_watermark_detection_result = detect(decoded_output_with_watermark, 
                                                    args, 
                                                    model,
                                                    device=device, 
                                                    tokenizer=tokenizer)
            
            if without_watermark_detection_result == None or with_watermark_detection_result == None:
                invaild_inputs_num += 1
                continue
            
            text_output_without_watermark, _ = without_watermark_detection_result
            print("text_output_without_watermark: ", text_output_without_watermark)
            for key, value in text_output_without_watermark:
                if key == "Prediction":
                    num_type1_error += 1 if value == "Watermarked" else 0
            text_output_with_watermark, _ = with_watermark_detection_result
            print("text_output_with_watermark: ", text_output_with_watermark)
            for key, value in text_output_with_watermark:
                if key == "Prediction":
                    num_type2_error += 1  if value == "Human/Unwatermarked" else 0
                    
            all_watermarked_outputs.append(decoded_output_with_watermark)
            all_unwatermarked_outputs.append(decoded_output_without_watermark)
            
        watermark_results = perplexity.compute(
            model_id='gpt2',
            predictions=all_watermarked_outputs
        )
            
        unwatermarked_results = perplexity.compute(
            model_id='gpt2',
            predictions=all_unwatermarked_outputs
        )
        
        print("invaild_inputs_num: ", invaild_inputs_num)
            
        print("num_type1_error: ", num_type1_error)
        print("num_type2_error: ", num_type2_error)
        
        print("Average Watermarked PPL:", watermark_results['mean_perplexity'])
        print("Average Unwatermarked PPL:", unwatermarked_results['mean_perplexity'])
        
    return
    

if __name__ == "__main__":

    args = parse_args()
    print(args)
    
    test_cnn_dailymail(args)
    
    

