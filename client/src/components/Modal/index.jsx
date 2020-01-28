import React from 'react';

import Backdrop from 'components/Backdrop';

import styles from './index.module.css';

const modal = props => {
    const classes = [styles.Modal];
    classes.push(props.isShown ? 'Shown' : )

    return (
        <>
            <Backdrop click={props.close} isShown={props.isShown}/>
            <div className={styles.Modal}>
                {props.children}
            </div>
        </>
    );
};

export default modal;