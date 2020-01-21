import React from 'react';

import styles from './index.module.css';

const slider = props => (
    <div className={styles.SliderContainer}>
        <label htmlFor={props.name}>{props.name}</label>
        <input
            type="range"
            id={props.name}
            min={props.min}
            max={props.max}
            step={props.step}
            className={styles.Slider}
            value={props.value}
            onChange={props.change}
        />
    </div>
);

export default slider;