import React from 'react';

import styles from './index.module.css';

const spinner = props => (
    <div className={`${styles.Spinner} ${styles[props.theme]}`}>Loading...</div>
);

export default spinner;